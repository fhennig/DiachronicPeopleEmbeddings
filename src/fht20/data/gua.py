"""Stuff that deals with the guardian data."""
import json
import gzip
import os
import psycopg2.extras
from datetime import datetime
from functools import cached_property
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Dict, Iterator
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait

from fht20.db import DBInfo
import fht20.data.util as util


class Doc:
    """A document, with all its meta data."""
    
    def __init__(self, raw: dict, parsed_doc: str, merged_mentions):
        self.raw = raw
        self.parsed_doc = parsed_doc
        self.merged_mentions = merged_mentions

    @property
    def id(self) -> str:
        return self.raw['id']

    @property
    def type(self) -> str:
        return self.raw['type']

    @cached_property
    def publication_date(self) -> datetime:
        d = self.raw['webPublicationDate']
        return datetime.strptime(d, '%Y-%m-%dT%H:%M:%SZ')

    @property
    def has_text(self) -> bool:
        return self.raw['fields']['bodyText'] != ''

    @property
    def text(self) -> Optional[str]:
        text = self.raw['fields']['bodyText']
        return text if text != '' else None

    @property
    def tokens(self) -> Optional[str]:
        """The tokens in the text, seperated by whitespace."""
        text = self.text
        if text is None:
            return None
        return " ".join([t.text for t in self.parsed_doc])

    @property
    def lowercased_tokens(self) -> List[str]:
        if not self.has_text:
            return []
        return [t.text.lower().replace('\x00', '')
                for t in self.parsed_doc]

    @property
    def token_count(self) -> int:
        if not self.has_text:
            return 0
        return len(self.parsed_doc)

    @property
    def section_id(self) -> str:
        return self.raw['sectionId']

    @property
    def section_name(self) -> str:
        return self.raw['sectionName']

    @property
    def pillar_id(self) -> Optional[str]:
        return self.raw.get('pillarId', None)

    @property
    def pillar_name(self) -> Optional[str]:
        return self.raw.get('pillarName', None)


class Mention:
    """A mention of a person, generated from a Document."""
    def __init__(self, doc_id: str, tokens: str,
                 start_index: int, end_index: int):
        """Takes the document id, the start and end index in tokens,
        as well as the tokens, seperated by white space.""" 
        self.doc_id = doc_id
        self.tokens = tokens
        self.start_index = start_index
        self.end_index = end_index
        self.parent_id = None
        self.person_id = None

    def __repr__(self):
        return f"Mention({self.tokens}, {self.start_index}, {self.end_index})"


class DocumentExtractor:
    def __init__(self, nlp):
        """Takes an nlp object to parse text."""
        self.nlp = nlp
        self.excluded = set(["'s", "’s"])

    def _get_mentions(self, doc_id, processed_text) -> List[Tuple[List[str], List[Mention]]]:
        mentions = []
        for ent in processed_text.ents:
            if ent.label_ != 'PERSON':
                continue
            # TODO filter also 'Mr' and 'Lady' and 'Baroness'
            tokens = " ".join([token.text for token in ent
                               if token.text not in self.excluded])
            mention = Mention(doc_id, tokens, ent.start, ent.end)
            mentions.append(mention)
        # merge the mentions
        people = []
        for mention in mentions:
            handled = False
            names = mention.tokens.split()
            for person_names, person_mentions in people:
                if all([name in person_names for name in names]):
                    person_mentions.append(mention)
                    handled = True
                    break
            if not handled:
                people.append((names, [mention]))
        return people

    def json_to_doc(self, json_data) -> Doc:
        raw_text = json_data['fields']['bodyText']
        if raw_text == '':
            return Doc(json_data, None, [])
        processed_text = self.nlp(raw_text)
        mentions = self._get_mentions(json_data['id'], processed_text)
        return Doc(json_data, processed_text, mentions)


class DocToDBLoader:
    def __init__(self, nlp, db):
        self.extractor = DocumentExtractor(nlp)
        self.db = db

    def load(self, json_data):
        doc = self.extractor.json_to_doc(json_data)
        self.db.insert_document_object(doc)
    

class PersMention:
    """A Mention loaded from the DB, thus includes an ID."""
    def __init__(self, id, doc_id, start_index, end_index, raw, parent_id, preprocessing_type):
        self.id = id
        self.doc_id = doc_id
        self.start_index = start_index
        self.end_index = end_index
        self.raw = raw
        self.parent_id = parent_id
        self.preprocessing_type = preprocessing_type

    def __repr__(self):
        return f"PersMention({self.id}, {self.raw})"


class GuardianCorpus:
    """Represents a corpus of documents from the Guardian OpenPlatform."""
    
    def __init__(self, data_dir: str):
        """The corpus is initialized from a directory path."""
        self.data_dir = data_dir

    def files_iter(self, year=None, month=None):
        """An iterator through the files in the corpus, the files are gzip
        compressed."""
        for filename in sorted(os.listdir(self.data_dir)):
            if year is not None:
                if not filename.startswith(str(year)):
                    continue
            if month is not None:
                if f"-{month:02}." not in filename:
                    continue
            path = os.path.join(self.data_dir, filename)
            yield path

    def load_docs(self, path):
        with gzip.open(path, 'r') as f:
            return json.load(f)

    def json_doc_iter(self, year=None, month=None):
        """An iterator through the documents in the corpus.
        Yields raw json documents."""
        for path in self.files_iter(year=year, month=month):
            for json_doc in self.load_docs(path):
                yield json_doc


def merge_person_tokens(mentions: List[Dict[str, Any]], tokens: List[str]) -> List[str]:
    """Takes a list of mentions and embeds them in the token list."""
    for mention in reversed(mentions):
        tokens[mention['start']:mention['end']] = [mention['person_name']]
    return tokens


class GuaDB:
    """A class that wraps a database cursor."""
    def __init__(self, db_info: DBInfo, script_dir: str, pool_size=4):
        self.db_info = db_info
        self.script_dir = script_dir
        self.pool = db_info.get_connection_pool(pool_size)
        self.conn = db_info.get_connection()
        self.person_cache = {}
        self.vocab_cache = {}

    def run_fn(self, cursor_fn):
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            res = cursor_fn(cursor)
        self.pool.putconn(conn)
        return res

    def _load_script(self, script_name: str) -> str:
        script_path = os.path.join(self.script_dir, f'{script_name}.sql')
        with open(script_path, 'r') as f:
            return f.read()

    def _exec_script(self, script_name: str):
        script = self._load_script(script_name)
        conn = self.pool.getconn()
        with conn.cursor() as cur:
            cur.execute(script)
        conn.commit()
        self.pool.putconn(conn)

    def init_schema(self):
        self._exec_script('init_db')

    def add_indices(self):
        """Adds a bunch of indices to the data for faster queries.
        The indices are added in parallel over multiple connections."""
        index_creation_queries = [
#            "create unique index unique_words on guardian_vocab(word);",
            "create index mentions_doc_id_index on guardian_mentions(doc_id);"
#            "create index tokens_doc_id_index on guardian_doc_tokens(doc_id);"
        ]
        executor = ThreadPoolExecutor(max_workers=len(index_creation_queries))
        futures = []
        for query in index_creation_queries:
            def f():
                conn = self.pool.getconn()
                with conn.cursor() as cur:
                    cur.execute(query)
                conn.commit()
                self.pool.putconn(conn)
            futures.append(executor.submit(f))
        wait(futures)

    def init_person_cache(self):
        with self.conn.cursor() as cursor:
            cursor.execute("select raw, id from guardian_persons;")
            self.person_cache = dict(cursor.fetchall())

    def init_vocab_cache(self):
        with self.conn.cursor() as cursor:
            cursor.execute("select word, id from guardian_vocab;")
            self.vocab_cache = dict(cursor.fetchall())

    def insert_document(self, conn, doc: Doc):
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into guardian_documents(id, doc, type, date, has_text, tokens,
                            token_count, section_id, section_name, pillar_id, pillar_name)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                (doc.id, json.dumps(doc.raw), doc.type, doc.publication_date, doc.has_text, list(doc.lowercased_tokens),
                 doc.token_count, doc.section_id, doc.section_name, doc.pillar_id, doc.pillar_name)
            )

    def insert_word(self, conn, word: str):
        cache_hit = self.vocab_cache.get(word, None)
        if cache_hit:
            return cache_hit
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into guardian_vocab(word)
                values (%s)
                on conflict(word) do update set word = excluded.word
                returning id;""",
                (word,)
            )
            id = cursor.fetchone()[0]
        self.vocab_cache[word] = id
        return id

    def insert_doc_tokens(self, conn, doc: Doc):
        if not doc.has_text:
            return
        token_ids = [self.insert_word(conn, word)
                     for word in doc.lowercased_tokens]
        rows = [(doc.id, i, token_id) for i, token_id
                in enumerate(token_ids)]
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(cursor,
                """
                insert into guardian_doc_tokens(doc_id, token_index, token_id)
                values %s;""",
                rows
            )

    def insert_mention(self, conn, mention: Mention):
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into guardian_mentions(doc_id, start_index, end_index,
                            raw, parent_id, preprocessing_type, person_id)
                values (%s, %s, %s, %s, %s, %s, %s)
                returning id
                ;""",
                (mention.doc_id, mention.start_index, mention.end_index,
                 mention.tokens, mention.parent_id, "DEFAULT", mention.person_id)
            )
            return cursor.fetchone()[0]

    def insert_person(self, conn, name) -> int:
        """Inserts a person into the person table, if it does not exist yet.
        In either case, returns the ID of the person."""
        # try to lookup in the cache
        cache_hit = self.person_cache.get(name, None)
        if cache_hit:
            return cache_hit
        # no cache hit, do the insert
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into guardian_persons(raw)
                values (%s)
                on conflict(raw) do update set raw = excluded.raw
                returning id
                ;""",
                (name,)
            )
            id = cursor.fetchone()[0]
        # put it in the cache
        self.person_cache[name] = id
        # return id
        return id

    def insert_persons(self, conn, names) -> List[int]:
        """Inserts the names into the DB and returns a list of ids matching the names"""
        assert len(set(names)) == len(names)
        intermediate_result = []
        to_insert = []
        for name in names:
            cache_hit = self.person_cache.get(name, None)
            if cache_hit:
                intermediate_result.append(('retrieved', cache_hit))
            else:
                intermediate_result.append(('new', len(to_insert)))
                to_insert.append((name,))
        if not to_insert:
            assert len(intermediate_result) == len(names)
            return [id for _, id in intermediate_result]
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(cursor,
                """
                insert into guardian_persons(raw)
                values %s
                returning id;""",
                to_insert,
                page_size=len(to_insert)+1
            )
            new_ids = [r[0] for r in cursor.fetchall()]
            assert len(new_ids) == len(to_insert), f"new_ids: {len(new_ids)}, to_insert: {len(to_insert)}: {to_insert}"
            for name, id in zip(to_insert, new_ids):
                self.person_cache[name[0]] = id
        result = []
        for t, number in intermediate_result:
            if t == 'retrieved':
                result.append(number)
            elif t == 'new':
                result.append(new_ids[number])
        return result

    def insert_document_object(self, doc: Doc):
        """Inserts a whole document object, including mentions etc."""
        conn = self.conn
        try:
            # insert document
            self.insert_document(conn, doc)
            # insert individual tokens
            self.insert_doc_tokens(conn, doc)
            # insert persons
            names = [mnts[0].tokens for names, mnts in doc.merged_mentions]
            ids = self.insert_persons(conn, names)
            for i, v in enumerate(doc.merged_mentions):
                _, mnts = v
                for mnt in mnts:
                    mnt.person_id = ids[i]
            # insert and mentions
            for _, mnts in doc.merged_mentions:
                first_id = self.insert_mention(conn, mnts[0])
                for sub_mention in mnts[1:]:
                    sub_mention.parent_id = first_id
                    self.insert_mention(conn, sub_mention)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def get_document_ids(self, year, month):
        start, end = util.get_range_for_month(year, month)
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select id from guardian_documents
                where date between %s and %s
                ;""",
                (start, end)
            )
            res = [row[0] for row in cursor.fetchall()]
        self.pool.putconn(conn)
        return res

    def get_all_doc_ids(self):
        with self.conn.cursor() as cursor:
            cursor.execute("select id from guardian_documents;")
            return [r[0] for r in cursor.fetchall()]

    def get_all_doc_ids_with_text(self):
        with self.conn.cursor() as cursor:
            cursor.execute("select id from guardian_documents where has_text = true;")
            return [r[0] for r in cursor.fetchall()]

    def get_mentions(self, doc_id) -> List[PersMention]:
        result = []
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select id, doc_id, start_index, end_index, raw, parent_id, preprocessing_type
                from guardian_mentions
                where doc_id = %s
                ;""",
                (doc_id,)
            )
            for row in cursor.fetchall():
                mention = PersMention(*row)
                result.append(mention)
        self.pool.putconn(conn)
        return result

    def update_mentions(self, start_date, end_date, mapping):
        """Used to change what person a mention refers to in a specific time
        frame.  In 2000, 'Clinton' probably refers to 'Bill Clinton'."""
        conn = self.pool.getconn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                update guardian_mentions
                set person_id = v.new_id
                from (values %s) as v(old_id, new_id)
                where doc_id in (select gd.id
                                 from guardian_documents gd
                                 where gd.date > '{start_date}'
                                   and gd.date < '{end_date}')
                  and person_id = v.old_id;
                """,
                mapping.items(),
                page_size=300
            )
        conn.commit()
        self.pool.putconn(conn)

    def get_vocabulary(self, year=None, month=None) -> List[Tuple[str, int]]:
        """Returns a list of tuples, words map to word counts.  Optionally a
        year and month can be given, to restrict the count to a
        specific time frame. The result is ordered by occurence count,
        descending."""
        year_clause, month_clause = "", ""
        if year:
            year_clause = f"and extract(year from gd.date) = {year}"
        if month:
            month_clause = f"and extract(month from gd.date) = {month}"
        def query(cursor):
            cursor.execute(
            f"""
            select v.word, count(*) as count
            from guardian_vocab v,
                 guardian_doc_tokens dt,
                 guardian_documents gd
            where v.id = dt.token_id
            and gd.id = dt.doc_id
            {year_clause}
            {month_clause}
            group by v.id
            order by count desc;""")
            return cursor.fetchall()
        return self.run_fn(query)

    def get_persons_vocab(self, year=None, month=None) -> List[Tuple[str, int]]:
        """Like get_vocabulary, except that it returns people and their
        occurence counts."""
        year_clause, month_clause = "", ""
        if year:
            year_clause = f"and extract(year from gd.date) = {year}"
        if month:
            month_clause = f"and extract(month from gd.date) = {month}"
        def query(cursor):
            cursor.execute(
            f"""
            select gp.raw, count(*) as count
            from guardian_persons gp,
                 guardian_documents gd,
                 guardian_mentions gm
            where gm.person_id = gp.id
            and gm.doc_id = gd.id
            {year_clause}
            {month_clause}
            group by gp.raw
            order by count desc;""")
            return cursor.fetchall()
        return self.run_fn(query)

    def get_document_count(self, time_slice) -> int:
        s, e = time_slice
        with self.conn.cursor() as cur:
            cur.execute(f"select count(*) from guardian_documents where date > '{s}' and date < '{e}' and has_text;")
            return cur.fetchone()[0]

    def get_doc_tokens_without_mentions(self, time_slice=None) -> Iterator[List[str]]:
        def query(cursor):
            time_slice_clause = ""
            if time_slice:
                time_slice_clause = f"and gd.date > '{time_slice[0]}' and gd.date < '{time_slice[1]}'"
            cursor.execute(
                f"""
                select gd.id,
                       gd.tokens
                from guardian_documents gd
                where true
                  {time_slice_clause};""")
        conn = self.pool.getconn()
        with conn.cursor() as cur:
            query(cur)
            while True:
                rows = cur.fetchmany()
                if not rows:
                    break
                for doc_id, tokens in rows:
                    yield tokens
        self.pool.putconn(conn)

    def get_doc_tokens_with_mentions(self, time_slice=None) -> Iterator[List[str]]:
        def query(cursor):
            time_slice_clause = ""
            if time_slice:
                time_slice_clause = f"and gd.date > '{time_slice[0]}' and gd.date < '{time_slice[1]}'"
            cursor.execute(
                f"""
                select gd.id,
                       gd.tokens,
                       json_agg(json_build_object('start', gm.start_index,
                                                  'end', gm.end_index,
                                                  'person_name', gp.raw)
                                order by gm.start_index) as mentions
                from guardian_documents gd,
                     guardian_mentions gm,
                     guardian_persons gp
                where gm.doc_id = gd.id
                  and gm.person_id = gp.id
                  {time_slice_clause}
                group by gd.id;""")
        conn = self.pool.getconn()
        with conn.cursor() as cur:
            query(cur)
            while True:
                rows = cur.fetchmany()
                if not rows:
                    break
                for doc_id, tokens, mentions in rows:
                    yield merge_person_tokens(mentions, tokens)
        self.pool.putconn(conn)

    def insert_wikidata_objs(self, objs):
        conn = self.pool.getconn()
        rows = [(o.id, o.label_en, o.aliases_en, o.raw_json) for o in objs]
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(
                cursor,
                """
                insert into wikidata_objects(object_id, label_en, aliases_en, raw_object)
                values %s;
                """,
                rows
            )
        conn.commit()
        self.pool.putconn(conn)

    def delete_person_from_match_string(self, match_str):
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                delete from guardian_persons
                where raw similar to %s;
                """,
                (match_str,)
            )
        conn.commit()    
        self.pool.putconn(conn)

    def persons_with_section_counts(self, year, min_count):
        """Returns a dictionary mapping people to dictionaries of section: count.
        A min_count is given, which gives the minimum mention count a person needs
        to have to be included."""
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select person,
                       json_agg(json_build_object(section, count) order by count desc)
                from (select gp.raw        as person,
                             gd.section_id as section,
                             count(*)      as count
                      from guardian_persons gp,
                           guardian_mentions gm,
                           guardian_documents gd
                      where extract(year from gd.date) = %s
                        and gd.id = gm.doc_id
                        and gm.person_id = gp.id
                      group by gd.section_id, gp.raw) as t
                group by person
                having sum(count) >= %s
                order by sum(count) desc;
                """,
                (year, min_count)
            )
            res = {r[0]: util.list_of_dicts_to_dict(r[1])
                   for r in cursor.fetchall()}
        self.pool.putconn(conn)
        return res

    def documents_for_year_with_section(self, year):
        """Returns a list of tuples: (section_id, tokens)"""
        conn = self.pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select gd.section_id, gd.tokens
                from guardian_documents gd
                where extract(year from gd.date) = %s;
                """,
                (year,)
            )
            res = cursor.fetchall()
        self.pool.putconn(conn)
        return res
