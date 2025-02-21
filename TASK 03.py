import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website to scrape
URL = "http://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_books(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    book_list = []
    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text
        stock = book.find("p", class_="instock availability").text.strip()
        book_list.append({"Title": title, "Price": price, "Availability": stock})

    return book_list


# Scrape data
books_data = get_books(URL)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(books_data)
df.to_csv("books.csv", index=False)

print("Data saved to books.csv")
