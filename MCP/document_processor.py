#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific governing permissions and
# limitations under the License.

import os
import json
import fitz
import docx
import pptx
from MCP import mcp
from pathlib import Path
from pptx import Presentation
from typing import Dict, Any, List, Optional

class DocumentProcessor:
    """
    Advanced document processing for PDF, DOCX, PPTX files with MCP integration.
    Provides text extraction, structure analysis, and metadata extraction.
    """
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': 'PDF Document',
            '.docx': 'Word Document',
            '.doc': 'Word 97-2003 Document',
            '.pptx': 'PowerPoint Presentation',
            '.ppt': 'PowerPoint 97-2003 Presentation'
        }
    
    def _validate_file(self, file_path: str) -> Path:
        """Validate file path and check if it exists."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return path
    
    def extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content from PDF file."""
        try:
            path = self._validate_file(file_path)
            doc = fitz.open(str(path))
            
            content = {
                "pages": [],
                "metadata": doc.metadata,
                "total_pages": len(doc),
                "file_size": path.stat().st_size
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract tables and images
                blocks = page.get_text("dict")
                tables = self._extract_tables_from_page(blocks)
                images = self._extract_images_from_page(page)
                
                content["pages"].append({
                    "page_number": page_num + 1,
                    "text": text,
                    "tables": tables,
                    "images": images,
                    "word_count": len(text.split())
                })
            
            doc.close()
            return {
                "success": True,
                "format": "pdf",
                "content": content,
                "summary": {
                    "total_words": sum(p["word_count"] for p in content["pages"]),
                    "total_tables": sum(len(p["tables"]) for p in content["pages"]),
                    "total_images": sum(len(p["images"]) for p in content["pages"])
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_docx_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content from DOCX file."""
        try:
            path = self._validate_file(file_path)
            doc = docx.Document(str(path))
            
            content = {
                "paragraphs": [],
                "tables": [],
                "metadata": {
                    "core_properties": {
                        "title": doc.core_properties.title,
                        "author": doc.core_properties.author,
                        "subject": doc.core_properties.subject,
                        "created": str(doc.core_properties.created),
                        "modified": str(doc.core_properties.modified)
                    }
                },
                "file_size": path.stat().st_size
            }
            
            # Extract paragraphs
            for para in doc.paragraphs:
                content["paragraphs"].append({
                    "text": para.text,
                    "style": para.style.name if para.style else "Normal",
                    "alignment": str(para.alignment) if para.alignment else None
                })
            
            # Extract tables
            for table_idx, table in enumerate(doc.tables):
                table_data = {
                    "table_index": table_idx,
                    "rows": [],
                    "column_count": len(table.columns),
                    "row_count": len(table.rows)
                }
                
                for row in table.rows:
                    row_data = []
                    for cell in row.cells:
                        row_data.append(cell.text)
                    table_data["rows"].append(row_data)
                
                content["tables"].append(table_data)
            
            return {
                "success": True,
                "format": "docx",
                "content": content,
                "summary": {
                    "total_paragraphs": len(content["paragraphs"]),
                    "total_tables": len(content["tables"]),
                    "total_words": sum(len(p["text"].split()) for p in content["paragraphs"])
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_pptx_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content from PPTX file."""
        try:
            path = self._validate_file(file_path)
            prs = Presentation(str(path))
            
            content = {
                "slides": [],
                "metadata": {
                    "slide_width": prs.slide_width,
                    "slide_height": prs.slide_height,
                    "slide_count": len(prs.slides)
                },
                "file_size": path.stat().st_size
            }
            
            for slide_idx, slide in enumerate(prs.slides):
                slide_content = {
                    "slide_number": slide_idx + 1,
                    "title": "",
                    "text_content": [],
                    "tables": [],
                    "notes": slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
                }
                
                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        if shape.text_frame and shape.text_frame.paragraphs:
                            first_para = shape.text_frame.paragraphs[0]
                            if first_para.level == 0 and not slide_content["title"]:
                                slide_content["title"] = shape.text
                            else:
                                slide_content["text_content"].append({
                                    "text": shape.text,
                                    "level": first_para.level if first_para else 0
                                })
                    
                    # Extract tables
                    if shape.has_table:
                        table = shape.table
                        table_data = {
                            "rows": [],
                            "column_count": len(table.columns),
                            "row_count": len(table.rows)
                        }
                        
                        for row in table.rows:
                            row_data = [cell.text for cell in row.cells]
                            table_data["rows"].append(row_data)
                        
                        slide_content["tables"].append(table_data)
                
                content["slides"].append(slide_content)
            
            return {
                "success": True,
                "format": "pptx",
                "content": content,
                "summary": {
                    "total_slides": len(content["slides"]),
                    "total_tables": sum(len(s["tables"]) for s in content["slides"]),
                    "total_words": sum(
                        len(text["text"].split()) 
                        for slide in content["slides"] 
                        for text in slide["text_content"]
                    )
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_tables_from_page(self, blocks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables from PDF page blocks."""
        tables = []
        # Simplified table extraction - can be enhanced
        return tables
    
    def _extract_images_from_page(self, page) -> List[Dict[str, Any]]:
        """Extract images from PDF page."""
        images = []
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            images.append({
                "image_index": img_idx,
                "width": img[2],
                "height": img[3],
                "colorspace": img[5],
                "size": len(img[7]) if len(img) > 7 else 0
            })
        
        return images
    
    def get_document_summary(self, file_path: str) -> Dict[str, Any]:
        """Get quick summary of document without full content extraction."""
        try:
            path = self._validate_file(file_path)
            extension = path.suffix.lower()
            
            if extension == '.pdf':
                doc = fitz.open(str(path))
                summary = {
                    "format": "pdf",
                    "pages": len(doc),
                    "metadata": doc.metadata,
                    "file_size": path.stat().st_size
                }
                doc.close()
            elif extension == '.docx':
                doc = docx.Document(str(path))
                summary = {
                    "format": "docx",
                    "paragraphs": len(doc.paragraphs),
                    "tables": len(doc.tables),
                    "metadata": {
                        "title": doc.core_properties.title,
                        "author": doc.core_properties.author
                    },
                    "file_size": path.stat().st_size
                }
            elif extension == '.pptx':
                prs = Presentation(str(path))
                summary = {
                    "format": "pptx",
                    "slides": len(prs.slides),
                    "file_size": path.stat().st_size
                }
            else:
                return {"success": False, "error": f"Unsupported format: {extension}"}
            
            return {"success": True, "summary": summary}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize document processor
document_processor = DocumentProcessor()

@mcp.tool()
def extract_document_content(file_path: str, include_full_content: bool = True) -> Dict[str, Any]:
    """
    Extract content from PDF, DOCX, or PPTX files.
    
    Args:
        file_path: Path to the document file
        include_full_content: Whether to include full content or just summary
    
    Returns:
        Dictionary containing extracted content and metadata
    """
    if not include_full_content:
        return document_processor.get_document_summary(file_path)
    
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return document_processor.extract_pdf_content(file_path)
    elif extension in ['.docx', '.doc']:
        return document_processor.extract_docx_content(file_path)
    elif extension in ['.pptx', '.ppt']:
        return document_processor.extract_pptx_content(file_path)
    else:
        return {
            "success": False,
            "error": f"Unsupported format: {extension}",
            "supported_formats": list(document_processor.supported_formats.keys())
        }

@mcp.tool()
def list_supported_formats() -> Dict[str, Any]:
    """List all supported document formats."""
    return {
        "success": True,
        "formats": document_processor.supported_formats,
        "description": "Supported document formats for processing"
    }

@mcp.tool()
def batch_process_documents(directory_path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    Process multiple documents in a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        recursive: Whether to search subdirectories
    
    Returns:
        List of processed documents with summaries
    """
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            return {"success": False, "error": "Directory not found"}
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in document_processor.supported_formats:
                result = document_processor.get_document_summary(str(file_path))
                if result["success"]:
                    documents.append({
                        "file_path": str(file_path),
                        "summary": result["summary"]
                    })
        
        return {
            "success": True,
            "directory": str(path),
            "documents": documents,
            "count": len(documents)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}