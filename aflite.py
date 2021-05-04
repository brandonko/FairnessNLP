from tqdm import tqdm
import pandas as pd
import sys
import os
import numpy as np
import logging

# Minimum final dataset size
N = 15000

# Iterations (refinement) for predictability scores
M = 5

# Number of deletions to make per elimination round
k = 250

# Training Filepath
tweets_train_fp = './our_data/original/train.tsv'
out_filepath = f'./our_data/aflite/{k}k_{M}M_{N}N/'
if not os.path.exists(out_filepath):
    os.mkdir(out_filepath)

def aflite():
    # Import training data
    train_csv = pd.read_csv(tweets_train_fp, sep='\t', header=None)
    train_processed_data = []

    for i, row in train_csv.iterrows():
        processed_row = [row[2], 1 if row[1] == 'positive' else 0, row[0]]
        train_processed_data.append(processed_row)

    df = pd.DataFrame(train_processed_data, columns=['sentence', 'sentiment', 'id'])
    # print(train_processed_df.head())

    # df = pd.DataFrame.from_dict({
    #     "sentence": ['big good first sentence.', 'very bad second', 'good good third', 'good four', 'good five', 'and six'],
    #     "sentiment": [1, 0, 1, 1, 1, 1],
    #     "id": [111, 222, 333, 444, 555, 666]
    # })
    print('\n\n----statistics----')
    print('len of training df:', len(df))
    print('k (eliminations per round):', k)
    print('M (iterations):', M)
    print('N (min dataset size):', N)

    iteration = 1
    continue_flag = True
    while continue_flag:
        print(f'\n\n---AFLITE ITERATION {iteration}----\n\n')
        iteration += 1
        predictability_scores = {}
        for i in tqdm(range(M)):
            
            # Random train dev split
            train = df.sample(frac = 0.5, random_state=np.random.RandomState())
            dev = df.drop(train.index)

            # print(train)
            # print(dev)

            results = train_and_eval(train, dev)

            for p, l, sent, id in results:
                if id not in predictability_scores:
                    predictability_scores[id] = 0
                if p == l:
                    predictability_scores[id] += 1
        
        ids = []
        scores = []
        for id, score in predictability_scores.items():
            ids.append(id)
            scores.append(score)
        
        pred_scores_df = pd.DataFrame.from_dict({
            'id': ids,
            'scores': scores
        })

        pred_scores_df = pred_scores_df.sort_values(by=['scores'], ascending=False)
        for i, row in pred_scores_df.iterrows():
            df = df[df['id'] != row['id']]
            if i > k:
                break
            if len(df) <= N:
                continue_flag = False
                break
        
        # Save results of predictability scores
        preds_filename = f'pred_scores.tsv'
        if not os.path.isfile(os.path.join(out_filepath, preds_filename)):
            pred_scores_df.to_csv(os.path.join(out_filepath, preds_filename), header=False, sep = '\t')
        
        # print('\n\n---preds---\n\n', pred_scores_df)
        # print('\n\n---df---\n\n', df)

    file_name = f'train_aflite.tsv'
    df.to_csv(os.path.join(out_filepath, file_name), header=False, sep = '\t')

# Hyperparams

LEARNING_RATE = 2e-5
EPSILON = 1e-8
EPOCHS = 4

# Need to re-run block 3.4 if BATCH_SIZE is changed
BATCH_SIZE = 32

import time
import copy
import torch
import sklearn
import datetime
import random
import numpy as np

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup
)
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader, RandomSampler, 
    SequentialSampler
)
from sklearn.metrics import f1_score

device = torch.device("cuda")

def train_and_eval(train_df, dev_df):
    logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
    # Get the lists of sentences and their labels.
    sentences = train_df.sentence.values
    labels = train_df.sentiment.values

    dev_sentences = dev_df.sentence.values  
    dev_labels = dev_df.sentiment.values
    dev_ids = dev_df.id.values

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, Warning=None)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    dev_input_ids = []
    dev_attention_masks = []

    # For every sentence...
    for sent in dev_sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        dev_input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        dev_attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    dev_input_ids = torch.cat(dev_input_ids, dim=0)
    dev_attention_masks = torch.cat(dev_attention_masks, dim=0)
    dev_labels = torch.tensor(dev_labels)

    # Combine the training inputs into a TensorDataset.
    val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = BATCH_SIZE

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                shuffle = False,
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = LEARNING_RATE, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = EPSILON # args.adam_epsilon  - default is 1e-8.
                    )
    
    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = EPOCHS

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # Save the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        # print("")
        # print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        # print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 250 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)     

            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        # print("")
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # print("")
        # print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Validation results for f1 score calculation
        y_trues = []
        y_preds = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the 
            # softmax.
            loss = result.loss
            logits = result.logits
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            y_preds.extend(np.argmax(logits, axis=1).flatten())
            y_trues.extend(label_ids.flatten())

            
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Calculate F1
        # Calculate F1
        f1 = f1_score(y_trues, y_preds)
        # print("  F1 Score: {0:.2f}".format(f1))
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'F1 Score': f1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        # deep copy the model
        if avg_val_accuracy > best_acc:
            best_epoch = epoch_i + 1
            best_acc = avg_val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    # print("")
    # print("Training complete!")

    # print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    # print("Best validation score:", best_acc, "at epoch number:", best_epoch)
    model.load_state_dict(best_model_wts)

    # Re evaluate on dev

    model.eval()
    # Validation results for f1 score calculation
    y_trues = []
    y_preds = []
    y_sentences = []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = result.loss
        logits = result.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

        y_preds.extend(np.argmax(logits, axis=1).flatten())
        y_trues.extend(label_ids.flatten())
        for in_ids in b_input_ids:
            y_sentences.append(tokenizer.decode(in_ids).replace(tokenizer.pad_token, '').strip())
    
    return zip(y_preds, y_trues, y_sentences, dev_ids)

if __name__ == '__main__':
    aflite()