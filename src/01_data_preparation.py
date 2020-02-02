from services.scrape import ScrapeService
from services.clean import CleanService
from services.deduplication import DeduplicationService
from settings import SCRAP_DATASET_ROOT

if __name__ == '__main__':
    ScrapeService(SCRAP_DATASET_ROOT).scrape()
    CleanService(SCRAP_DATASET_ROOT).scan()
    deduplication = DeduplicationService(SCRAP_DATASET_ROOT)
    deduplication.scan()
    deduplication.save()
