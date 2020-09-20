#! /usr/bin/env python3
"""This file contains argument parsing and calls functions from the
fht20 modules.  In there is most of the code to do the actual stuff.
"""
import logging
import os
import re
import argparse
import spacy
from typing import List, Tuple
from dotenv import load_dotenv
from threading import Thread
from queue import Queue
from tqdm import tqdm

from fht20.data.guardian_api_crawler import GuardianApiCrawler
from fht20.train_data_builder import train_data_builder_from_str
from fht20.model_trainer import trainer_from_str
from fht20.doc_puller import DocPuller
import fht20.data.gua as gua
import fht20.db as db

def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

initialize_logging()
logger = logging.getLogger()

load_dotenv()


def multithreading_guardian_inserting(corpus, n_threads, year, add_indices):
    """Inserts documents that are read from file into the database.
    Currently works with multiple processing threads that do the NLP
    processing, and a single database thread that does the inserting
    of objects.
    """
    class DBWorker(Thread):
        def __init__(self, db, doc_queue):
            super(DBWorker, self).__init__()
            self.doc_queue = doc_queue
            self.db = db

        def run(self):
            logger.debug("DB Worker started.")
            for doc in iter(self.doc_queue.get, None):
                try:
                    self.db.insert_document_object(doc)
                except Exception as e:
                    print("Error inserting document {doc.id}")
                    print(e)
                    print("SKIPPING!")
                    
            logger.debug("DB Worker terminated.")

    class ProcessingWorker(Thread):
        def __init__(self, json_data_queue, doc_queue):
            super(ProcessingWorker, self).__init__()
            self.json_data_queue = json_data_queue
            self.doc_queue = doc_queue
            # load nlp model
            nlp = spacy.load("en_core_web_sm")
            nlp.disable_pipes('tagger', 'parser')
            self.extractor = gua.DocumentExtractor(nlp)

        def run(self):
            logger.debug("Processing Worker started.")
            for json_data in iter(self.json_data_queue.get, None):
                doc = self.extractor.json_to_doc(json_data)
                self.doc_queue.put(doc)
            self.doc_queue.put(None)
            logger.debug("Processing Worker Terminated.")
    
    # open db
    logger.info("Opening DB connection ...")
    db_info = db.DBInfo.load_from_env()
    gua_db = gua.GuaDB(db_info, 'sql_src/gua')
    logger.info("Filling DB cache:")
    logger.info("  Person cache ...")
    gua_db.init_person_cache()
    person_cache_count = len(gua_db.person_cache)
    logger.info(f"  {person_cache_count:,} persons cached.")
    logger.info("  Vocab cache ...")
    gua_db.init_vocab_cache()
    vocab_cache_count = len(gua_db.vocab_cache)
    logger.info(f"  {vocab_cache_count:,} words cached.")
    logger.info("  Document cache ...")
    existing_doc_ids = set(gua_db.get_all_doc_ids())
    logger.info(f"  {len(existing_doc_ids):,} document ids cached.")

    logger.info("Starting processes ...")
    j_q = Queue(maxsize=100)
    d_q = Queue(maxsize=100)
    for _ in range(n_threads):
        p_w = ProcessingWorker(j_q, d_q)
        p_w.start()
    d_w = DBWorker(gua_db, d_q)
    d_w.start()

    logger.info("Starting processing and inserting!")
    with tqdm() as pbar:
        for doc in corpus.json_doc_iter(year):
            doc_id = doc['id']
            if doc_id in existing_doc_ids:
                continue
            existing_doc_ids.add(doc_id)
            j_q.put(doc)
            pbar.set_description(f"j_q: {j_q.qsize()}, d_q: {d_q.qsize()}")
            pbar.update()
        for _ in range(n_threads):
            j_q.put(None)

    if add_indices:
        gua_db.add_indices()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Code for the paper TODO TODO. "
            "There are 7 commands which are explained below in the order they should be used. "
            "The steps are roughly: Downloading articles from the OpenPlatfrom API, loading the "
            "articles into a database, processing them and then dumping the text files again. "
            "Then creating PPMI matrices for specific time slices and lastly building a model for "
            "a set of slices."
        )
    )
    subparsers = parser.add_subparsers(title="command", dest="command")

    dl = subparsers.add_parser(
        "download",
        help=("Download a time range of articles from the guardian API. "
              "Results are stored in zip files, one file  per month.")
    )
    dl.add_argument('from_year', type=int)
    dl.add_argument('from_month', type=int)
    dl.add_argument('to_year', type=int)
    dl.add_argument('to_month', type=int)
    dl.add_argument('directory', type=str)

    gua_init = subparsers.add_parser(
        "init-schema",
        help=("Initializes the schema (tables) for the guardian data. "
              "Tables for documents, mentions, persons, vocabulary are created.")
    )

    load_gua = subparsers.add_parser(
        "load",
        help=("Load a downloaded corpus into a database. "
              "This loads the previously created zip files into the database.")
    )
    load_gua.add_argument('data_dir', type=str, help="The corpus directory.")
    load_gua.add_argument('--year', type=int, help="Only load a specific year.")
    load_gua.add_argument('--proc', type=int, default=4)
    load_gua.add_argument('--add_indices', default=False, action='store_true')

    gua_indices = subparsers.add_parser(
        "add-indices",
        help=("Add indices to the tables. "
              "This will make subsequent queries faster, but would be slower during insertions.")
    )

    get_base_data = subparsers.add_parser(
        "get-base-data",
        help=("Creates base data, which is just a dump of text from the database. "
              "It is used to subsequently create the PPMI matrices with 'create-ppmi-slices'.")
    )
    get_base_data.add_argument('dir', type=str)
    get_base_data.add_argument('start_year', type=int)
    get_base_data.add_argument('start_month', type=int)
    get_base_data.add_argument('end_year', type=int)
    get_base_data.add_argument('end_month', type=int)
    get_base_data.add_argument('--n_workers', type=int, default=4)
    get_base_data.add_argument('--no_mentions', default=True, action='store_false')

    create_training_data = subparsers.add_parser(
        "create-ppmi-slices",
        help=("Creates training data based on base data. "
              "The data_specifier specifies the amount and length of time slices, as well as "
              "the window size used for mutual information scores.  The vocabulary is created "
              "based on a given minimum occurrence count.")
    )
    create_training_data.add_argument(
        'data_specifier', type=str,
        help=("The directory name, which also specifies the parameters of the data generation. "
              "The format is: {prefix}_{start_year}-{start_month}_{slice_count}x{slice_size}_w{window_size}m{min_count}."))
    create_training_data.add_argument('-P', '--n_processes',
                                      type=int, default=4,
                                      help="Number of parallel processes to use.")

    train = subparsers.add_parser(
        "train",
        help="Train a model"
    )
    train.add_argument(
        "model_path",
        type=str,
        help=("The path where the model should be saved.  Should be {train_data_dir}/models/{model_dir}.  "
              "model_dir should have the format: {model_name}_{embed_dim}_l{lambda}t{tau}g{gamma}.")
    )
    train.add_argument(
        "epochs",
        type=int,
        help=("Up to how many epochs should be done.  If '10' is given and there are already 5, "
              "5 more will be executed.")
    )
    train.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="The size of a single batch."
    )
    
    args = parser.parse_args()

    if args.command == "download":
        logger.info("Download selected.  Initializing ...")
        crawler = GuardianApiCrawler()
        crawler.download(args.from_year, args.from_month,
                         args.to_year, args.to_month,
                         args.directory)
        logger.info("Download finished.")

    elif args.command == "init-schema":
        logger.info("Loading DB access info ...")
        db_info = db.DBInfo.load_from_env()
        logger.info("Connecting to DB ...")
        db = gua.GuaDB(db_info, 'sql_src/gua')
        logger.info("Initializing schema ...")
        db.init_schema()
        logger.info("Done.")

    elif args.command == "load":
        logger.info(f"Processing data with {args.proc} processes.")
        if args.add_indices:
            logger.info(f"Indices will be added right after processing.")
        corpus = gua.GuardianCorpus(args.data_dir)
        multithreading_guardian_inserting(corpus, args.proc, args.year, args.add_indices)
        logger.info("Done.")

    elif args.command == "add-indices":
        logger.info("Loading DB access info ...")
        db_info = db.DBInfo.load_from_env()
        logger.info("Connecting to DB ...")
        db = gua.GuaDB(db_info, 'sql_src/gua')
        logger.info("Indexing loaded data ...")
        db.add_indices()
        logger.info("Done.")

    elif args.command == "get-base-data":
        logger.info("Loading DB access info ...")
        db_info = db.DBInfo.load_from_env()
        logger.info("Connecting to DB ...")
        db = gua.GuaDB(db_info, 'sql_src/gua', args.n_workers)
        logger.info("Starting processing ...")
        dp = DocPuller(db, args.dir, args.n_workers,
                       args.start_year, args.start_month,
                       args.end_year, args.end_month,
                       args.no_mentions)
        dp.process()
        logger.info("Done.")

    elif args.command == "create-ppmi-slices":
        logger.info("Extracting training data ...")
        p1, p2 = args.data_specifier.split('/slices/')
        data_dir = os.path.join(p1, 'base_data')
        train_data_builder = train_data_builder_from_str(p2, data_dir, args.data_specifier)
        train_data_builder.collect_all(args.n_processes)
        logger.info("Done.")

    elif args.command == "train":
        train_path, model_str = args.model_path.split("/models/")
        args.epochs
        args.batch_size
        trainer = trainer_from_str(train_path, model_str, args.batch_size)
        trainer.run(args.epochs)
