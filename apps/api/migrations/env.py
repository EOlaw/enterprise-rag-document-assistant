"""Read the connection URL from KIP_DATABASE_URL; never commit credentials here."""

from logging.config import fileConfig

from alembic import context
from kip.db.models import Base
from kip.settings import Settings
from sqlalchemy import create_engine, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations() -> None:
    if context.is_offline_mode():
        context.configure(
            url=Settings().database_url.get_secret_value(),
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    # Tests can supply an already-open connection (including its isolated schema).
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        context.configure(
            connection=supplied_connection, target_metadata=target_metadata, compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(Settings().database_url.get_secret_value(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata, compare_type=True
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


run_migrations()
