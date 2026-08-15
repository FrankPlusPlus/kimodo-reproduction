# Kimodo 汇报 HTML PPT

**不要只在 Cursor 编辑器里看 `index.html` 源码**——那是代码，不是幻灯片。要用浏览器打开渲染后的页面。

## 打开方式

本地服务已可这样起：

```bash
cd /home/yezitao/PublicWorkspace/yzt/kimodo-reproduction/docs/ppt
python3 -m http.server 8765
```

然后在本机浏览器访问：

- http://127.0.0.1:8765/
- 或 http://127.0.0.1:8765/index.html

也可：

```bash
xdg-open /home/yezitao/PublicWorkspace/yzt/kimodo-reproduction/docs/ppt/index.html
```

操作：`←` `→` 翻页 · `Home` / `End` · `F` 全屏 · 点击左右半屏翻页 · URL `#页码` 可直达。

图源：`assets/` 来自官方文档静态资源（`arch.png`、`overview.png`、`constraints.png` 等）。
