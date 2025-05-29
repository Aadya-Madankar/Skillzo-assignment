# file_manager.py
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

class FileManager:
    def __init__(self, storage_dir="stored_files"): 
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._ensure_metadata_file_exists()

    def _ensure_metadata_file_exists(self):
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def save_file_with_metadata(self, 
                                file_name: str, 
                                temp_pdf_source_path: str,
                                temp_vector_store_source_path: str, 
                                expiry_days=7) -> str:
        timestamp = int(datetime.now().timestamp())
        # Sanitize file_name for file_id to prevent path traversal, use stem for cleaner ID
        safe_file_name_part = "".join(c if c.isalnum() or c in ('_') else '_' for c in Path(file_name).stem)
        file_id = f"{timestamp}_{safe_file_name_part}"
        
        expiry_date = datetime.now() + timedelta(days=expiry_days)
        
        file_dir = self.storage_dir / file_id
        file_dir.mkdir(exist_ok=True)
        
        # Save the actual PDF to its permanent location
        permanent_pdf_path = file_dir / Path(file_name).name # Keep original filename within its dedicated folder
        try:
            if Path(temp_pdf_source_path).exists():
                shutil.copy(temp_pdf_source_path, permanent_pdf_path)
                print(f"PDF '{file_name}' copied to '{permanent_pdf_path}'")
            else:
                print(f"Warning: Temporary PDF source path not found: {temp_pdf_source_path}")
                if file_dir.exists(): shutil.rmtree(file_dir) # Cleanup
                raise IOError(f"Temporary PDF source path not found: {temp_pdf_source_path}")
        except Exception as e:
            print(f"Error copying PDF from temp to permanent storage: {e}")
            if file_dir.exists(): shutil.rmtree(file_dir) # Cleanup
            raise IOError(f"Failed to store PDF '{file_name}' due to: {e}") from e
        
        permanent_vector_store_path = file_dir / "faiss_index" # Standardized name
        
        if Path(temp_vector_store_source_path).exists() and Path(temp_vector_store_source_path, "index.faiss").exists():
            shutil.copytree(temp_vector_store_source_path, permanent_vector_store_path, dirs_exist_ok=True)
            print(f"Vector store for '{file_name}' copied to '{permanent_vector_store_path}'")
        else:
            print(f"Warning: Temporary vector store source path not found or incomplete: {temp_vector_store_source_path}")
            if file_dir.exists(): shutil.rmtree(file_dir) # Cleanup if vector store fails
            raise IOError(f"Temporary vector store for '{file_name}' not found or incomplete: {temp_vector_store_source_path}")
            
        metadata = self.load_metadata()
        metadata[file_id] = {
            "original_name": Path(file_name).name,
            "upload_date": datetime.now().isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "pdf_path": str(permanent_pdf_path), # Ensure this is saved
            "vector_store_path": str(permanent_vector_store_path)
        }
        self.save_metadata(metadata)
        
        return file_id
    
    def load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    content = f.read()
                    if not content.strip(): # Handle empty file case
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.metadata_file}. Returning empty metadata.")
                return {}
            except Exception as e:
                print(f"Error loading metadata: {e}. Returning empty metadata.")
                return {}
        return {}
    
    def save_metadata(self, metadata: Dict[str, Any]):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def get_active_files(self) -> Dict[str, Any]:
        metadata = self.load_metadata()
        active_files = {}
        current_time = datetime.now()
        
        for file_id, info in metadata.items():
            try:
                expiry_date = datetime.fromisoformat(info["expiry_date"])
                if current_time < expiry_date:
                    # Check if vector store (specifically index.faiss) AND pdf_path exist and are valid
                    if "vector_store_path" in info and Path(info["vector_store_path"], "index.faiss").exists() and \
                       "pdf_path" in info and Path(info["pdf_path"]).exists():
                        active_files[file_id] = info
                    else:
                        print(f"Warning: Vector store or PDF for active file_id {file_id} not found or path invalid. VS='{info.get('vector_store_path')}', PDF='{info.get('pdf_path')}'. Skipping.")
            except (ValueError, KeyError) as e:
                print(f"Error parsing metadata for file_id {file_id}: {e}. Skipping.")
                continue
        
        return active_files

    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        active_files = self.get_active_files()
        return active_files.get(file_id)
    
    def cleanup_expired_files(self) -> int:
        metadata = self.load_metadata()
        current_time = datetime.now()
        updated_metadata = {}
        cleaned_count = 0
        
        for file_id, info in list(metadata.items()): # Use list() for safe iteration while deleting
            try:
                expiry_date = datetime.fromisoformat(info["expiry_date"])
                if current_time < expiry_date:
                    # Before keeping, ensure its directory and index/pdf still exist
                    if "vector_store_path" in info and Path(info["vector_store_path"], "index.faiss").exists() and \
                       "pdf_path" in info and Path(info["pdf_path"]).exists():
                         updated_metadata[file_id] = info
                    else:
                        print(f"Info: Vector store or PDF for active file_id {file_id} missing during cleanup. Removing record and directory.")
                        file_dir_to_clean = self.storage_dir / file_id 
                        if file_dir_to_clean.exists():
                            shutil.rmtree(file_dir_to_clean)
                        cleaned_count +=1 
                else: # File is expired
                    file_dir = self.storage_dir / file_id
                    if file_dir.exists():
                        shutil.rmtree(file_dir)
                    cleaned_count += 1
            except (ValueError, KeyError) as e:
                print(f"Error processing expiry for file_id {file_id}: {e}. Marking for removal.")
                file_dir = self.storage_dir / file_id 
                if file_dir.exists():
                    shutil.rmtree(file_dir)
                cleaned_count += 1 
        
        self.save_metadata(updated_metadata)
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} expired/invalid file(s)/record(s).")
        return cleaned_count
    
    def delete_file(self, file_id: str) -> bool:
        metadata = self.load_metadata()
        
        if file_id in metadata:
            file_dir = self.storage_dir / file_id 
            if file_dir.exists():
                try:
                    shutil.rmtree(file_dir)
                    print(f"Successfully deleted directory: {file_dir}")
                except Exception as e:
                    print(f"Error deleting directory {file_dir}: {e}")
            
            del metadata[file_id]
            self.save_metadata(metadata)
            print(f"Successfully deleted metadata for file_id: {file_id}")
            return True
        
        print(f"File_id {file_id} not found in metadata for deletion.")
        return False