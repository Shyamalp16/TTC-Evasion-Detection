-- Database initialization script
-- This script runs when the PostgreSQL container starts for the first time

-- Create database if it doesn't exist (already handled by POSTGRES_DB env var)
-- CREATE DATABASE snitchsystem;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance
-- (Tables will be created by SQLAlchemy, but we can add custom indexes here)

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE snitchsystem TO snitch;
