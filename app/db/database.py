import socket

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# SQLite requires check_same_thread=False for FastAPI
connect_args = {}
database_url = make_url(settings.DATABASE_URL)

if database_url.drivername.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

if database_url.drivername.startswith("postgresql"):
    hostaddr = settings.DATABASE_HOSTADDR
    if not hostaddr and settings.DATABASE_FORCE_IPV4 and database_url.host:
        try:
            addr_info = socket.getaddrinfo(
                database_url.host,
                database_url.port or 5432,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror:
            addr_info = []
        if addr_info:
            hostaddr = addr_info[0][4][0]

    if hostaddr and "hostaddr" not in database_url.query:
        connect_args["hostaddr"] = hostaddr

# Enhanced connection pool configuration for resilience
pool_config = {
    "pool_pre_ping": True,  # Test connections before using
    "connect_args": connect_args,
}

# Add PostgreSQL-specific pool settings
if database_url.drivername.startswith("postgresql"):
    pool_config.update({
        "pool_size": settings.DATABASE_POOL_SIZE,
        "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        "pool_recycle": settings.DATABASE_POOL_RECYCLE,
        "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
        "pool_pre_ping": True,  # Validate connections before use
    })

engine = create_engine(
    settings.DATABASE_URL,
    **pool_config,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
