import requests
from bs4 import BeautifulSoup

def get_headlines(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  headlines = []
  for headline in soup.find_all('h3', class_='title'):
    headlines.append(headline.text)
  return headlines

if __name__ == '__main__':
  url = 'https://www.nasdaq.com/market-activity/stocks/tsla/news-headlines'
  headlines = get_headlines(url)

  for headline in headlines:
    print(headline)
