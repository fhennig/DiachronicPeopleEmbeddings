create table if not exists guardian_documents
(
	id text
		constraint guardian_documents_pk
			primary key,
	doc text not null,
	type text not null,
	date timestamp not null,
	has_text boolean not null,
	tokens text[] not null,
	token_count int not null,
	section_id text not null,
	section_name text not null,
	pillar_id text,
	pillar_name text
);

create table if not exists guardian_mentions
(
    id serial primary key,
    doc_id text not null,
    start_index int not null,
    end_index int not null,
    raw text not null,
    parent_id int,
    preprocessing_type text,
    person_id int
);

create table if not exists guardian_persons
(
    id serial primary key,
    raw text not null
);

create unique index guardian_persons_raw_uindex
	on guardian_persons (raw);

create table if not exists guardian_vocab
(
    id serial primary key,
    word text unique not null
);

create table if not exists guardian_doc_tokens
(
    doc_id text not null,
    token_index int not null,
    token_id int not null
);

