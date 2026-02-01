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

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
