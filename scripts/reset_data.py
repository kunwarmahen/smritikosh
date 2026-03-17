#!/usr/bin/env python3
"""
reset_data.py — wipe all user data across PostgreSQL, Neo4j, and MongoDB.

Drops every row from every data table (preserving schema and app_users).
Use --include-users to also wipe the app_users table.
Use --user <username> to wipe data for a single user only.

Usage:
    python scripts/reset_data.py                        # wipe all data, keep users
    python scripts/reset_data.py --include-users        # wipe everything including users
    python scripts/reset_data.py --user alice           # wipe data for alice only
    python scripts/reset_data.py --dry-run              # show what would be deleted
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

# ── Config from env ────────────────────────────────────────────────────────────

POSTGRES_URL  = os.getenv("POSTGRES_URL", "")
NEO4J_URI     = os.getenv("NEO4J_URI",    "bolt://localhost:7687")
NEO4J_USER    = os.getenv("NEO4J_USER",   "neo4j")
NEO4J_PASS    = os.getenv("NEO4J_PASSWORD", "")
MONGODB_URL   = os.getenv("MONGODB_URL",  "")
MONGODB_DB    = os.getenv("MONGODB_DB_NAME", "smritikosh_audit")


# ── PostgreSQL ─────────────────────────────────────────────────────────────────

async def reset_postgres(user: str | None, include_users: bool, dry_run: bool) -> None:
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
    except ImportError:
        print("  [skip] sqlalchemy not installed")
        return

    if not POSTGRES_URL:
        print("  [skip] POSTGRES_URL not set")
        return

    engine = create_async_engine(POSTGRES_URL)

    # Tables with user_id — can filter directly
    user_tables = [
        "memory_feedback",
        "user_beliefs",
        "user_facts",
        "user_procedures",
    ]
    # memory_links has no user_id — linked via event UUIDs
    # events must be deleted last (FK target for memory_links/memory_feedback)

    async with engine.begin() as conn:
        tag = f"user={user}" if user else "all"

        for table in user_tables:
            if user:
                result = await conn.execute(text(f"DELETE FROM {table} WHERE user_id = :uid"), {"uid": user})
            else:
                result = await conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            print(f"  {'[DRY] ' if dry_run else ''}postgres  {table:<20} ({tag}) → {result.rowcount} rows")

        # memory_links: delete via subquery on events
        if user:
            result = await conn.execute(text(
                "DELETE FROM memory_links WHERE from_event_id IN "
                "(SELECT id FROM events WHERE user_id = :uid) "
                "OR to_event_id IN (SELECT id FROM events WHERE user_id = :uid)"
            ), {"uid": user})
        else:
            result = await conn.execute(text("TRUNCATE TABLE memory_links CASCADE"))
        print(f"  {'[DRY] ' if dry_run else ''}postgres  {'memory_links':<20} ({tag}) → {result.rowcount} rows")

        # events last
        if user:
            result = await conn.execute(text("DELETE FROM events WHERE user_id = :uid"), {"uid": user})
        else:
            result = await conn.execute(text("TRUNCATE TABLE events CASCADE"))
        print(f"  {'[DRY] ' if dry_run else ''}postgres  {'events':<20} ({tag}) → {result.rowcount} rows")

        if dry_run:
            await conn.rollback()
            return

        if include_users and not user:
            await conn.execute(text("TRUNCATE TABLE app_users CASCADE"))
            print("  postgres  app_users            (all) → wiped")

    await engine.dispose()


# ── Neo4j ──────────────────────────────────────────────────────────────────────

async def reset_neo4j(user: str | None, dry_run: bool) -> None:
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        print("  [skip] neo4j driver not installed")
        return

    if not NEO4J_URI or not NEO4J_PASS:
        print("  [skip] NEO4J_URI / NEO4J_PASSWORD not set")
        return

    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    async with driver.session() as session:
        if user:
            # Count first
            count_res = await session.run(
                "MATCH (n {user_id: $uid}) RETURN count(n) AS c", uid=user
            )
            count = (await count_res.single())["c"]

            if not dry_run:
                await session.run(
                    "MATCH (n {user_id: $uid}) DETACH DELETE n", uid=user
                )
            print(f"  {'[DRY] ' if dry_run else ''}neo4j     nodes/rels            (user={user}) → {count} nodes")
        else:
            count_res = await session.run("MATCH (n) RETURN count(n) AS c")
            count = (await count_res.single())["c"]

            if not dry_run:
                await session.run("MATCH (n) DETACH DELETE n")
            print(f"  {'[DRY] ' if dry_run else ''}neo4j     nodes/rels            (all) → {count} nodes")

    await driver.close()


# ── MongoDB ────────────────────────────────────────────────────────────────────

async def reset_mongodb(user: str | None, dry_run: bool) -> None:
    try:
        import motor.motor_asyncio as motor
    except ImportError:
        print("  [skip] motor not installed")
        return

    if not MONGODB_URL:
        print("  [skip] MONGODB_URL not set")
        return

    client = motor.AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB]

    collections = await db.list_collection_names()
    for col_name in collections:
        col = db[col_name]
        filt = {"user_id": user} if user else {}
        count = await col.count_documents(filt)

        tag = f"user={user}" if user else "all"
        print(f"  {'[DRY] ' if dry_run else ''}mongodb   {col_name:<20} ({tag}) → {count} docs")

        if not dry_run:
            if user:
                await col.delete_many(filt)
            else:
                await col.drop()

    client.close()


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe Smritikosh data across all databases.")
    parser.add_argument("--user",          metavar="USERNAME", help="Wipe data for one user only")
    parser.add_argument("--include-users", action="store_true", help="Also wipe the app_users table (full reset)")
    parser.add_argument("--dry-run",       action="store_true", help="Show counts without deleting")
    args = parser.parse_args()

    if args.include_users and args.user:
        print("Error: --include-users and --user cannot be combined.")
        sys.exit(1)

    scope = f"user={args.user}" if args.user else ("ALL data including users" if args.include_users else "ALL data")
    mode  = "DRY RUN — " if args.dry_run else ""

    print(f"\n{mode}Resetting Smritikosh: {scope}\n")

    if not args.dry_run:
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        print()

    await reset_postgres(args.user, args.include_users, args.dry_run)
    await reset_neo4j(args.user, args.dry_run)
    await reset_mongodb(args.user, args.dry_run)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.\n")


if __name__ == "__main__":
    asyncio.run(main())
