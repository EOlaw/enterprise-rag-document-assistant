"""Short-lived transactions; never hold a DB lock while parsing a document."""

from sqlite3 import Connection

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        # SQLite is only a lightweight test adapter, never the production queue.
        @event.listens_for(engine, "connect")
        def _foreign_keys(connection: object, _: object) -> None:
            assert isinstance(connection, Connection)
            connection.execute("PRAGMA foreign_keys=ON")

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
