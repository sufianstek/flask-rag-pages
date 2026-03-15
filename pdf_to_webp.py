#!/usr/bin/env python3
"""
PDF to WebP Converter

This script converts PDF files to WebP format using pdf2image and Pillow libraries.
Note: This requires the following packages to be installed:
- pdf2image
- Pillow

Usage:
    python3 pdf_to_webp.py

Example:
    Reads all PDF files under 'static/pdf/', renames unsafe filenames
    (for example, replaces spaces with underscores), and creates folders
    under 'static/pages/' such as 'static/pages/document1_pages'
    containing all pages as numbered WebP files.
"""

import sys
import os
import re
from pdf2image import convert_from_path


def make_safe_filename(name):
    """Return a filesystem-safe filename stem."""
    # Replace spaces first, then drop unsupported characters.
    safe = name.replace(" ", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "document"


def ensure_safe_pdf_filename(pdf_path):
    """Rename PDF to a safe filename if needed and return final path."""
    directory = os.path.dirname(pdf_path)
    base_name = os.path.basename(pdf_path)
    stem, ext = os.path.splitext(base_name)

    safe_stem = make_safe_filename(stem)
    safe_name = f"{safe_stem}{ext.lower()}"

    if safe_name == base_name:
        return pdf_path

    target_path = os.path.join(directory, safe_name)

    # Avoid collisions by appending a numeric suffix.
    if os.path.exists(target_path):
        counter = 1
        while True:
            candidate_name = f"{safe_stem}_{counter}{ext.lower()}"
            candidate_path = os.path.join(directory, candidate_name)
            if not os.path.exists(candidate_path):
                target_path = candidate_path
                break
            counter += 1

    os.rename(pdf_path, target_path)
    print(f"Renamed: {base_name} -> {os.path.basename(target_path)}")
    return target_path

def pdf_to_webp_folder(pdf_path, dpi=200):
    """
    Convert PDF to WebP format with all pages in a folder

    Args:
        pdf_path (str): Path to input PDF file
        dpi (int): DPI resolution for conversion (default: 200)
    """
    try:
        # Check if input file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")

        # Build output root under project static/pages
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_root = os.path.join(script_dir, "static", "pages")

        # Get PDF filename without extension
        filename = os.path.splitext(os.path.basename(pdf_path))[0]

        # Create folder name based on PDF filename
        folder_name = f"{filename}_pages"
        output_folder = os.path.join(output_root, folder_name)

        # Skip if folder already exists, assuming it was already converted.
        if os.path.isdir(output_folder):
            print(f"Skipping {pdf_path}: output folder already exists ({output_folder})")
            return None

        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created folder: {output_folder}")

        # Convert PDF to list of PIL Images
        print(f"Converting {pdf_path} to images...")
        pages = convert_from_path(pdf_path, dpi=dpi)

        if not pages:
            raise ValueError("No pages found in PDF file")

        # Save each page as WebP with numbered filenames
        print(f"Saving pages as WebP files in {output_folder}")
        for i, page in enumerate(pages):
            # Create filename with zero-padded page number
            webp_filename = f"{i+1}.webp"
            webp_path = os.path.join(output_folder, webp_filename)
            # Save with 30% quality
            page.save(webp_path, 'WEBP', quality=30)
            print(f"Saved: {webp_filename}")

        print(f"Successfully converted {len(pages)} pages from {pdf_path} to {output_folder}")

    except Exception as e:
        print(f"Error converting PDF to WebP: {e}")
        return False

    return True


def convert_all_pdfs(pdf_root=None, dpi=200):
    """Convert all PDFs in a folder and return a summary dictionary."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_pdf_root = pdf_root or os.path.join(script_dir, "static", "pdf")

    if not os.path.isdir(resolved_pdf_root):
        raise FileNotFoundError(f"PDF directory not found: {resolved_pdf_root}")

    pdf_files = [
        os.path.join(resolved_pdf_root, name)
        for name in sorted(os.listdir(resolved_pdf_root))
        if name.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return {
            "total": 0,
            "converted": 0,
            "skipped": 0,
            "failed": 0,
            "pdf_files": [],
            "pdf_root": resolved_pdf_root,
        }

    print(f"Found {len(pdf_files)} PDF files in {resolved_pdf_root}")

    # Ensure source filenames are safe before conversion.
    valid_pdf_files = [ensure_safe_pdf_filename(pdf_file) for pdf_file in pdf_files]

    success_count = 0
    skipped_count = 0
    failed_count = 0
    for pdf_file in valid_pdf_files:
        result = pdf_to_webp_folder(pdf_file, dpi=dpi)
        if result is True:
            success_count += 1
        elif result is None:
            skipped_count += 1
        else:
            failed_count += 1

    return {
        "total": len(valid_pdf_files),
        "converted": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "pdf_files": valid_pdf_files,
        "pdf_root": resolved_pdf_root,
    }

def main():
    """Convert all PDFs from static/pdf to static/pages."""
    try:
        summary = convert_all_pdfs()
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    if summary["total"] == 0:
        print(f"No PDF files found in {summary['pdf_root']}")
        sys.exit(1)

    print(
        f"Completed: converted {summary['converted']}, skipped {summary['skipped']}, "
        f"failed {summary['failed']}, total {summary['total']} PDF files"
    )

if __name__ == "__main__":
    main()