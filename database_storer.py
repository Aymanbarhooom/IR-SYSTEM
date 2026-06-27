# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:46:35 2026

@author: Ayman
"""
from services.document_store import DocumentStore

def main():
    store = DocumentStore()

    store.create_table()

    store.import_documents()
    
if __name__ == "__main__":
    main()