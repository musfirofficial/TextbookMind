from rag.parser import extract_student_materials

pages = extract_student_materials("data/history G-11 E.pdf")

for page in pages[34:35]:
    print(f"Page {page['page_number']}: \n {page['text']}")
