"""
Development server runner.
"""
import uvicorn
from app.main import app
from app.database import create_tables
from app.config import settings

if __name__ == "__main__":
    # Create database tables
    create_tables()
    
    # Run development server
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
