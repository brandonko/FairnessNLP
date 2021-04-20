import os
import pandas as pd
import math

compiled_file = open('twitter-all.csv', 'x')
seen_tweets = set()
for filename in os.listdir('./twitter-raw/'):
    tweet_file = open('./twitter-raw/' + filename)
    for line in tweet_file:
        parts = line.split('\t')
        # skip if duplicate or neutral tweet
        if parts[0] in seen_tweets or (parts[1] != 'positive' and parts[1] != 'negative'):
            continue
        seen_tweets.add(parts[0])
        
        # cut off \n character to account for last lines
        if parts[len(parts) - 1].endswith('\n'):
            last_part = parts[len(parts) - 1]
            parts[len(parts) - 1] = last_part[:len(last_part) - 1]

        # wrap tweet around quotes
        if parts[2].startswith('"') and parts[len(parts) - 1].endswith('"'):
            parts[2] = ''.join(parts[2:])
        else:
            parts[2] = '"' + ''.join(parts[2:]) + '"'
        parts = parts[:3]
        
        # escape quotes and replace tabs with spaces
        '''
        filtered = '"'
        for i in range(1, len(parts[2]) - 1):
            if parts[2][i] == '"':
                filtered += '\"'
            elif parts[2][i] == '\t':
                filtered += ' '
            else:
                filtered += parts[2][i]
        filtered += '"'
        parts[2] = filtered
        '''
    
        # write to file
        line = '\t'.join(parts) + '\n'
        compiled_file.write(line)
    tweet_file.close()
compiled_file.close()

# check for duplicates and neutrals
compiled_file = open('twitter-all.csv')
seen_tweets = set()
for line in compiled_file:
    parts = line.split('\t')
    if parts[0] in seen_tweets:
        print('duplicate tweet', parts[0])
    seen_tweets.add(parts[0])
    if parts[1] != 'positive' and parts[1] != 'negative':
        print('neutral tweet', parts[0])
