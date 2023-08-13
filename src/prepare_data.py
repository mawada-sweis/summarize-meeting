import json
import load_dataset
import remove_punctuation
import remove_stopwords
import tokanize_text
import var_operations
import tags
import stemming_text

MAIN_FILE_NAME = "transcript_summary.csv"
PUNCTUATION_CLEANED_FILE_NAME = 'punctuation_cleaned.csv'
TOKENS_FILE_NAME = 'tokenized_text.csv'
STOPWORDS_CLEANED_FILE_NAME = 'stopwords_cleaned_text.csv'
STEMMING_TOKENS_FILE_NAME = 'stemming_tokens.csv'

def main_function():
    print("Running the main function from prepare_data.py\n")
    MODEL_ID = var_operations.get_value_by_var_name('user_model_ID')
    
    # Load the main dataset if not already loaded
    if not load_dataset.is_dataset_loaded(file_name=MAIN_FILE_NAME):
        main_dataset = load_dataset.load_dataset_by_name(split_type='train')
        main_dataset = load_dataset.save_to_csv(
            dataset=main_dataset,
            dataset_name=MAIN_FILE_NAME,
            columns_name=['transcript', 'summary']
        )

    # Remove punctuation marks from texts
    if not load_dataset.is_dataset_loaded(file_name=PUNCTUATION_CLEANED_FILE_NAME):
        main_dataset = load_dataset.get_loaded_data(file_name=MAIN_FILE_NAME)

        max_main_tokens = tokanize_text.get_max_tokens_counts(
            main_dataset['transcript'].apply(
                lambda x: tokanize_text.tokenize(x, model_ID=MODEL_ID)
                )
        )
        var_operations.save_var('max_main_tokens', max_main_tokens)

        main_dataset['transcript_no_punctuation'] = \
            main_dataset['transcript'].apply(remove_punctuation.remove_from_str)

        load_dataset.save_to_csv(
            dataset=main_dataset,
            dataset_name=PUNCTUATION_CLEANED_FILE_NAME,
            columns_name=['transcript_no_punctuation', 'summary']
        )

    # Tokenize the texts
    if not load_dataset.is_dataset_loaded(file_name=TOKENS_FILE_NAME):
        punct_dataset = load_dataset.get_loaded_data(
            file_name=PUNCTUATION_CLEANED_FILE_NAME
        )

        punct_dataset['summary_tokens'] = punct_dataset['summary'].apply(
            lambda summary: tokanize_text.tokenize(summary, model_ID=MODEL_ID)
        )
        max_summary_tokens = \
            tokanize_text.get_max_tokens_counts(punct_dataset['summary_tokens'])
        var_operations.save_var('max_summary_tokens', max_summary_tokens)

        punct_dataset['transcript_tokens'] = \
            punct_dataset['transcript_no_punctuation'].apply(
            lambda trans: tokanize_text.tokenize(trans, model_ID=MODEL_ID)
        )
        max_transcript_tokens = \
            tokanize_text.get_max_tokens_counts(punct_dataset['transcript_tokens'])
        var_operations.save_var('max_transcript_tokens', max_transcript_tokens)
        punct_dataset['transcript_tokens'] = \
            punct_dataset['transcript_tokens'].apply(json.dumps)

        load_dataset.save_to_csv(
            dataset=punct_dataset,
            dataset_name=TOKENS_FILE_NAME,
            columns_name=['summary_tokens', 'transcript_tokens']
        )

    # Remove stopwords from texts
    if not load_dataset.is_dataset_loaded(file_name=STOPWORDS_CLEANED_FILE_NAME):
        tokens_dataset = load_dataset.get_loaded_data(file_name=TOKENS_FILE_NAME)

        tokens_dataset['transcript_tokens'] = \
            tokens_dataset['transcript_tokens'].apply(json.loads)

        tokens_dataset['clean_transcript'] = \
            tokens_dataset['transcript_tokens'].apply(
                remove_stopwords.remove_from_str_list
        )

        max_clean_transcript_tokens = tokanize_text.get_max_tokens_counts(
            tokens_dataset['clean_transcript']
        )
        var_operations.save_var('max_clean_transcript_tokens',
                                max_clean_transcript_tokens)

        tokens_dataset['clean_transcript'] = \
            tokens_dataset['clean_transcript'].apply(json.dumps)

        load_dataset.save_to_csv(
            dataset=tokens_dataset,
            dataset_name=STOPWORDS_CLEANED_FILE_NAME,
            columns_name=['clean_transcript']
        )

    if not load_dataset.is_dataset_loaded(file_name=STEMMING_TOKENS_FILE_NAME):
        stopwords_cleaned_dataset = load_dataset.get_loaded_data(file_name=STOPWORDS_CLEANED_FILE_NAME)
        stopwords_cleaned_dataset['clean_transcript'] = \
            stopwords_cleaned_dataset['clean_transcript'].apply(json.loads)

        stopwords_cleaned_dataset['stemmed_tokens'] = \
            [stemming_text.stemming_tokens(tokens_list) for tokens_list in stopwords_cleaned_dataset['clean_transcript']]

        max_stemmed_tokens = tokanize_text.get_max_tokens_counts(
            stopwords_cleaned_dataset['stemmed_tokens']
        )
        var_operations.save_var('max_stemmed_transcript_tokens',
                                max_stemmed_tokens)

        load_dataset.save_to_csv(
            dataset=stopwords_cleaned_dataset,
            dataset_name=STEMMING_TOKENS_FILE_NAME,
            columns_name=['stemmed_tokens']
        )

    # Get and save transcript tags if not already done
    if not var_operations.get_value_by_var_name('tags'):
        transcript = load_dataset.get_loaded_data(MAIN_FILE_NAME)['transcript']
        transcript_tags = tags.get_tags(transcript)
        var_operations.save_var('tags', transcript_tags)

    # Print token counts and tags
    print('The highest token count among the transcripts is: ',
        var_operations.get_value_by_var_name('max_main_tokens'))
    print('Maximum tokens in punctuation-free transcripts: ',
        var_operations.get_value_by_var_name('max_transcript_tokens'))
    print('Maximum tokens in clean transcripts: ',
        var_operations.get_value_by_var_name('max_clean_transcript_tokens'))
    print('Maximum stemmed tokens in clean transcripts:',
        var_operations.get_value_by_var_name('max_stemmed_transcript_tokens'))
    print('The highest token count among the summary is:',
        var_operations.get_value_by_var_name('max_summary_tokens'))
    print('The tags are:', var_operations.get_value_by_var_name('tags'))


if __name__ == "__main__":
    main_function()
