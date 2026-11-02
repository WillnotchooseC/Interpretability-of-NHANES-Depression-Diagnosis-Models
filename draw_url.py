"""
Coding Exercise: Decoding a Secret Message
--------------------------------------------
Given a published Google Doc URL, the document contains a table with:
    x-coordinate | Character | y-coordinate
Placing each character at its (x, y) position in a 2D grid and printing
it with a fixed-width font reveals a "secret message" made of uppercase
letters.

Usage:
    from decode_secret_message import print_secret_message
    print_secret_message("https://docs.google.com/document/d/xxxx/pub")
"""

import requests
from bs4 import BeautifulSoup


def print_secret_message(doc_url: str) -> None:
    """
    Take the URL of a published Google Doc, parse the coordinate table
    inside it, and print the resulting character grid (the secret
    message).

    :param doc_url: URL of the published Google Doc
    """
    grid = _parse_grid_data(doc_url)
    _print_grid(grid)


def _parse_grid_data(doc_url: str) -> dict:
    """
    Fetch the document and parse it into a {(x, y): char} dictionary.
    """
    response = requests.get(doc_url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError("No table found in the document. Check that the URL is correct and the doc is published.")

    rows = table.find_all("tr")

    coord_to_char = {}

    # The first row is the header (x-coordinate, Character, y-coordinate); skip it.
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        x_text = cells[0].get_text().strip()
        char_text = cells[1].get_text()  # don't strip yet, some chars could be mistaken for whitespace
        y_text = cells[2].get_text().strip()

        # Some Unicode characters (e.g. solid blocks) may pick up extra
        # whitespace/newlines when parsed; keep the first non-whitespace char.
        char_text = char_text.strip("\n\r\t ")
        if not char_text:
            continue

        try:
            x = int(x_text)
            y = int(y_text)
        except ValueError:
            continue  # skip rows whose coordinates aren't valid integers

        coord_to_char[(x, y)] = char_text[0]

    if not coord_to_char:
        raise ValueError("Parsed 0 valid coordinate entries. Check that the document format matches the example.")

    return coord_to_char


def _print_grid(coord_to_char: dict) -> None:
    """
    Compute the grid bounds from the coordinate dictionary and print it.
    (0, 0) is the top-left corner; x increases to the right, y increases downward.
    """
    max_x = max(x for x, y in coord_to_char)
    max_y = max(y for x, y in coord_to_char)

    for y in range(max_y + 1):
        line_chars = [coord_to_char.get((x, y), " ") for x in range(max_x + 1)]
        print("".join(line_chars))


if __name__ == "__main__":
    # Test with the example document from the exercise (prints the letter 'F')
    example_url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
    print_secret_message(example_url)