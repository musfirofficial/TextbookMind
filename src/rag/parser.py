import pymupdf

MIN_IMAGE_DIM = 100  # Filters out tiny UI icons, bullet graphics, and 1px lines
HEADER_MARGIN = 50
FOOTER_MARGIN = 50


def check_page_has_figures(page, clean_zone, min_dim: int = MIN_IMAGE_DIM) -> bool:
    """
    Lightweight O(1) dictionary check to identify if genuine diagrams/figures
    exist on the page without rendering pixel arrays or writing to disk.
    """
    image_list = page.get_images()
    if not image_list:
        return False

    for img in image_list:
        xref = img[0]
        pixel_w = img[2]
        pixel_h = img[3]

        # Ignore tiny icons and decorative lines
        if pixel_w >= min_dim and pixel_h >= min_dim:
            for r in page.get_image_rects(xref):
                if r.intersects(clean_zone):
                    return True
    return False


def sort_blocks_reading_order(blocks, clean_zone):
    """
    Sorts a mixed list of text and table blocks into natural human reading order:
    - Identifies full-width elements (titles, banners, spanning tables) anywhere on page.
    - Accurately classifies 1-column vs. 2-column layout even with spanning headers/tables at top.
    - Uses vertical band partitioning to prevent coordinate rounding loss.
    """
    if not blocks:
        return []

    mid_x = (clean_zone.x0 + clean_zone.x1) / 2.0
    content_width = clean_zone.width

    spanning_blocks = []
    column_blocks = []

    # 1. Isolate full-width elements (tables or headers spanning > 55% or crossing the mid corridor)
    for b in blocks:
        width = b[2] - b[0]
        is_spanning = (width > content_width * 0.55) or (
            b[0] < (mid_x - 30) and b[2] > (mid_x + 30)
        )
        if is_spanning:
            spanning_blocks.append(b)
        else:
            column_blocks.append(b)

    # 2. Divide non-spanning column blocks into left and right groups
    left_blocks = [b for b in column_blocks if ((b[0] + b[2]) / 2.0) < mid_x]
    right_blocks = [b for b in column_blocks if ((b[0] + b[2]) / 2.0) >= mid_x]

    # 3. Detect 2-column structure via vertical overlap in the column blocks
    has_two_columns = False
    for lb in left_blocks:
        for rb in right_blocks:
            # Overlap check along the Y-axis: both blocks share vertical elevation
            if max(lb[1], rb[1]) < min(lb[3], rb[3]):
                has_two_columns = True
                break
        if has_two_columns:
            break

    # If it's a single-column page, top-to-bottom sorting is sufficient
    if not has_two_columns:
        return sorted(blocks, key=lambda b: b[1])

    # 4. Two-Column Sequencing with Band Partitioning
    # Spanning blocks act as horizontal boundary cuts (handles top, middle, or bottom tables)
    ordered_blocks = []
    spanning_blocks = sorted(spanning_blocks, key=lambda b: b[1])
    unassigned_cols = list(column_blocks)

    for sp in spanning_blocks:
        sp_top = sp[1]

        # Extract all column blocks whose vertical midpoint sits above this spanning block
        current_band = [b for b in unassigned_cols if ((b[1] + b[3]) / 2.0) < sp_top]
        unassigned_cols = [b for b in unassigned_cols if b not in current_band]

        # Sequence band: Left Column first, then Right Column
        band_left = sorted(
            [b for b in current_band if ((b[0] + b[2]) / 2.0) < mid_x],
            key=lambda b: b[1],
        )
        band_right = sorted(
            [b for b in current_band if ((b[0] + b[2]) / 2.0) >= mid_x],
            key=lambda b: b[1],
        )

        ordered_blocks.extend(band_left)
        ordered_blocks.extend(band_right)
        ordered_blocks.append(sp)  # Place spanning table / header right in place

    # Process remaining column blocks sitting beneath the final spanning element
    rem_left = sorted(
        [b for b in unassigned_cols if ((b[0] + b[2]) / 2.0) < mid_x],
        key=lambda b: b[1],
    )
    rem_right = sorted(
        [b for b in unassigned_cols if ((b[0] + b[2]) / 2.0) >= mid_x],
        key=lambda b: b[1],
    )

    ordered_blocks.extend(rem_left)
    ordered_blocks.extend(rem_right)

    return ordered_blocks


def extract_student_materials(pdf_path: str):
    """
    Parses a PDF into an array of clean page records with inline Markdown tables,
    natural reading flow across multi-column pages, and figure presence flags.
    """
    doc = pymupdf.open(pdf_path)
    extracted_pages = []

    for p_index, page in enumerate(doc):
        page_number = p_index + 1

        # Non-destructive header/footer boundary
        clean_zone = pymupdf.Rect(
            page.rect.x0,
            page.rect.y0 + HEADER_MARGIN,
            page.rect.x1,
            page.rect.y1 - FOOTER_MARGIN,
        )

        # -------------------------------------------------------------
        # STEP 1: Lightweight Figure Detection
        # -------------------------------------------------------------
        has_figures = check_page_has_figures(page, clean_zone)

        # -------------------------------------------------------------
        # STEP 2: Extract Tables & Create Synthetic Table Blocks
        # -------------------------------------------------------------
        tables = page.find_tables(clip=clean_zone)
        table_blocks = []
        table_rects = []

        for t_idx, table in enumerate(tables):
            t_box = table.bbox
            t_rect = pymupdf.Rect(t_box)
            table_rects.append(t_rect)

            table_md = table.to_markdown().strip()
            if not table_md:
                continue

            # Conforms to PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
            synthetic_block = (
                t_box[0],
                t_box[1],
                t_box[2],
                t_box[3],
                table_md,
                f"table_{t_idx}",
                "table",
            )
            table_blocks.append(synthetic_block)

        # -------------------------------------------------------------
        # STEP 3: Extract Text & Deduplicate Against Tables
        # -------------------------------------------------------------
        raw_blocks = page.get_text("blocks", clip=clean_zone, sort=False)
        filtered_text_blocks = []

        for b in raw_blocks:
            b_text = b[4].strip()
            if not b_text:
                continue

            b_rect = pymupdf.Rect(b[:4])
            b_center = pymupdf.Point((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

            # Drop text block if its center or >50% of its area sits inside a table
            is_inside_table = False
            for t_rect in table_rects:
                intersect = b_rect & t_rect
                if not intersect.is_empty and (
                    t_rect.contains(b_center)
                    or intersect.get_area() > 0.5 * b_rect.get_area()
                ):
                    is_inside_table = True
                    break

            if not is_inside_table:
                filtered_text_blocks.append(b)

        # -------------------------------------------------------------
        # STEP 4: Combine & Sort Into Natural Reading Order
        # -------------------------------------------------------------
        combined_blocks = filtered_text_blocks + table_blocks
        ordered_blocks = sort_blocks_reading_order(combined_blocks, clean_zone)

        page_text = "\n\n".join(b[4].strip() for b in ordered_blocks)

        extracted_pages.append(
            {
                "document_name": pdf_path,
                "page_number": page_number,
                "has_figures": has_figures,
                "text": page_text,
            }
        )

    doc.close()
    return extracted_pages
