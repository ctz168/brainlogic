"""
类人脑双系统全闭环AI架构 - 维基百科无限知识引擎
Human-Like Brain Dual-System Full-Loop AI Architecture - Wikipedia Search Engine

集成维基百科搜索，作为海马体 CA3 区的外部存储扩展 (存算分离)。
"""

import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any

class WikipediaTool:
    def __init__(self):
        self.api_url = "https://zh.wikipedia.org/w/api.php" # 默认中文
        
    def search(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """搜索维基百科并返回摘要"""
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit
        }
        
        try:
            url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                
            results = []
            for item in data.get("query", {}).get("search", []):
                snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace('</span>', "")
                results.append({
                    "title": item.get("title"),
                    "snippet": snippet,
                    "pageid": item.get("pageid"),
                    "semantic_pointer": f"wiki:{item.get('pageid')}"
                })
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

    def get_content(self, pageid: int) -> str:
        """获取页面详情内容"""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": pageid,
            "exintro": True,
            "explaintext": True
        }
        
        try:
            url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                
            pages = data.get("query", {}).get("pages", {})
            for pid, content in pages.items():
                return content.get("extract", "")
        except:
            return ""
        return ""
