from apscheduler.schedulers.background import BackgroundScheduler
from scraper.aljazeera_scraper import run_scraper
import atexit


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=run_scraper, trigger="interval", hours=1)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
