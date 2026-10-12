-- Teleco-Tutor database schema (Supabase Postgres, or any Postgres 15+ with pgvector).
-- Idempotent: safe to run again. Run it once in the Supabase SQL editor.
--
-- Security model: the browser never talks to these tables. Only the backend does,
-- with the database owner / service credentials. Row Level Security is enabled with
-- no policies, so Supabase's public API (anon/authenticated roles) can read nothing.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;
create extension if not exists unaccent with schema extensions;
create extension if not exists pgcrypto with schema extensions;

-- unaccent() is not IMMUTABLE, so it can't be used in an index or generated column directly.
create or replace function public.tt_unaccent(text) returns text
  language sql immutable parallel safe strict
  set search_path = extensions, public
as $$ select extensions.unaccent('extensions.unaccent'::regdictionary, $1) $$;

-- Per-user settings. The Gemini key is stored encrypted by the backend (Fernet).
create table if not exists public.tt_profiles (
  user_id        uuid primary key,
  gemini_key_enc text,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);

-- One row per uploaded PDF. status drives the resumable indexing queue.
create table if not exists public.tt_documents (
  id              uuid primary key default extensions.gen_random_uuid(),
  user_id         uuid not null,
  course          text not null,
  name            text not null,              -- path inside the course, e.g. "Course/Chapter 1/chap1.pdf"
  storage_path    text not null,              -- object key in the storage bucket
  size_bytes      bigint,
  pages           int,
  status          text not null default 'pending'
                  check (status in ('pending', 'processing', 'ready', 'error')),
  message         text not null default '',
  ocr_pages       int not null default 0,
  chunk_count     int not null default 0,
  embedding_model text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  unique (user_id, course, name)
);
create index if not exists tt_documents_user_status on public.tt_documents (user_id, status);

-- Searchable chunks: one embedding + full-text vector per piece of a page.
create table if not exists public.tt_chunks (
  id          bigint generated always as identity primary key,
  user_id     uuid not null,
  document_id uuid not null references public.tt_documents (id) on delete cascade,
  course      text not null,
  page        int not null,
  chunk_index int not null,
  content     text not null,
  ocr         boolean not null default false,
  embedding   extensions.vector(768) not null,
  fts         tsvector generated always as (to_tsvector('simple', public.tt_unaccent(content))) stored
);
create index if not exists tt_chunks_user_course on public.tt_chunks (user_id, course);
create index if not exists tt_chunks_document on public.tt_chunks (document_id);
create index if not exists tt_chunks_fts on public.tt_chunks using gin (fts);

alter table public.tt_profiles  enable row level security;
alter table public.tt_documents enable row level security;
alter table public.tt_chunks    enable row level security;
