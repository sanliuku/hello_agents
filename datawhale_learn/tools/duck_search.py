import sys
from typing import List, Dict
from ddgs import DDGS


def search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    timeout: int = 10,
    proxy: str = "http://127.0.0.1:7897",
) -> List[Dict[str, str]]:
    try:
        with DDGS(timeout=timeout, proxy=proxy) as ddgs:
            results = list(ddgs.text(query, region=region, max_results=max_results))
        print(f"🔍 搜索 '{query}' 完成，共 {len(results)} 条结果")
        return results
    except Exception as e:
        print(f"❌ 搜索时发生错误: {e}")
        return []


# if __name__ == "__main__":
#     keyword = " ".join(sys.argv[1:]) or "Python 教程"
#     searchResults = search(keyword)

#     for index, item in enumerate(searchResults, start=1):
#         print(f"\n{index}. {item['title']}")
#         print(f"   链接: {item['href']}")
#         print(f"   摘要: {item['body']}")
