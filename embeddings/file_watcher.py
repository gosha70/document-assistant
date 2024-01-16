import time
import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.vectorstores import Chroma
from embeddings.unstructured.document_splitter import DocumentSplitter
from embeddings.embedding_database import add_file_content_to_db

class FileWatcher:
    def __init__(self, db: Chroma, document_splitter: DocumentSplitter, dir: str):
        self.observer = Observer()
        self.dir_to_watch = dir
        self.db = db
        self.document_splitter = document_splitter
    
    @staticmethod
    def run_file_watcher(db: Chroma, document_splitter: DocumentSplitter, temp_dir):
        file_watcher = FileWatcher(db=db, document_splitter=document_splitter, dir=temp_dir)
        file_watcher.run()            

    def run(self):
        event_handler = TempFileHandler(db=self.db, document_splitter=self.document_splitter)
        self.observer.schedule(event_handler, self.dir_to_watch, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()
        self.cleanup()

    def cleanup(self):
        # Clean up the directory when done
        try:
            shutil.rmtree(self.dir_to_watch)
            print(f"Temporary directory {self.dir_to_watch} has been deleted.")
        except Exception as e:
            print(f"Error deleting temporary directory {self.dir_to_watch}: {e}")

class TempFileHandler(FileSystemEventHandler):
    def __init__(self, db: Chroma, document_splitter: DocumentSplitter):
        self.db = db
        self.document_splitter = document_splitter

    def on_created(self, event):
        if event.is_directory:
            return None

        # Take any action here when a file is created.
        print(f"New file has been detected: {event.src_path}.")
        time.sleep(1) # Ensuring the file is fully saved.
        add_file_content_to_db(docs_db=self.db, document_splitter=self.document_splitter, file_name=event.src_path)

        # Delete the file after processing.
        os.remove(event.src_path)
        print(f"Deleted file - {event.src_path}.")
