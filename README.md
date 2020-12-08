# Diachronic People Embeddings

This repository accompanies the paper ["Diachronic Embeddings for
People in the News" (Hennig, Wilson;
2020)](https://www.aclweb.org/anthology/2020.nlpcss-1.19.pdf).

The data downloading, processing and model building is all in the
code.

## Pre-trained Embeddings

The pretrained embedding files are found in the
`pretrained_embeddings` directory.  The vocabulary is found in
`vocab.tsv`, each row contains a token and its total count over all 20
years, separated by a `\tab`.  The embeddings are in the `npz` files.
There are two files for each year, `u` and `v`.  The embeddings in
each can be used on their own, or `u` and `v` can be concatenated to
create embeddings with twice as many dimensions.

## Processing Pipeline

### Install & Setup

The requirements need to be installed, and the NER/tokenization model
needs to be downloaded:

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    
The API downloading requires an API key.  Some parts of the pipeline
rely on access to a PostgreSQL database.  API key and DB access
parameters can be set in environment variables or in an `.env` file:

    API_KEY=12345678-abcd-cdef-1234-a1b2c3d4e5f6
    DB_HOST=localhost
    DB_PORT=5432
    DB_USERNAME=my_db_user
    DB_PASSWORD=P4s5w0rd
    DB_NAME=dbnamehere

### Usage

The commandline tool is documented:

    ./src/main.py --help
    
There are 7 supported commands:

- `download`: Download a time range of articles from the guardian
   API. Results are stored in zip files, one file per month.
- `init-schema`: Initializes the schema (tables) for the guardian
  data. Tables for documents, mentions, persons, vocabulary are
  created.
- `load`: Load a downloaded corpus into a database. This loads the
  previously created zip files into the database.
- `add-indices`: Add indices to the tables. This will make subsequent
  queries faster, but would be slower during insertions.
- `get-base-data`: Creates base data, which is just a dump of text
  from the database. It is used to subsequently create the PPMI
  matrices with 'create-ppmi-slices'.
- `create-ppmi-slices`: Creates training data based on base data. The
  data_specifier specifies the amount and length of time slices, as
  well as the window size used for mutual information scores. The
  vocabulary is created based on a given minimum occurrence count.
- `train`: Train a model.

### File Structure

TODO





