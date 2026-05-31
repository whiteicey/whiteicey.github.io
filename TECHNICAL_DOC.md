# whiteicey.github.io — 技术架构与更新文档

> 最后更新：2026-05-31 | 维护者参考文档

---

## 目录

1. [技术架构概览](#1-技术架构概览)
2. [文件结构](#2-文件结构)
3. [设计系统](#3-设计系统)
4. [更新记录](#4-更新记录)
5. [已知问题与排查经验](#5-已知问题与排查经验)
6. [维护指南](#6-维护指南)

---

## 1. 技术架构概览

### 基础技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 静态生成器 | Jekyll | GitHub Pages 原生支持 |
| CSS 预处理 | LESS → CSS | 通过 Grunt 编译 |
| CSS 框架 | Bootstrap 3.3.2 | 响应式布局基础 |
| 图标 | Font Awesome 4.6.3 | CDN 加载 |
| 字体 | Google Fonts | Inter + Source Serif 4 + JetBrains Mono |
| JS | jQuery + 原生 JS | Bootstrap JS + 自定义 phase2.js |
| 评论 | Utterances | 基于 GitHub Issues |
| PWA | Service Worker + manifest.json | 离线支持 |
| 部署 | GitHub Pages | 自动部署 main 分支 |

### 页面结构

```
_layouts/default.html          ← 基础骨架（含 head + nav + footer）
├── _layouts/page.html         ← 首页、About、Tags 页
│   └── index.html             ← 首页（featured + grid 卡片）
│   └── about.html             ← 关于页
│   └── tags.html              ← 标签页
├── _layouts/post.html         ← 文章详情页
├── _layouts/keynote.html      ← Keynote 演讲页
└── 404.html                   ← 404 页面
```

### 核心功能模块

| 模块 | 实现文件 | 说明 |
|------|---------|------|
| 暗色模式 | `js/phase2.js` + `_includes/head.html` CSS Variables | localStorage 持久化 |
| 阅读进度条 | `js/phase2.js` + `css/hux-blog.css` `#reading-progress` | teal→gold 渐变 |
| 返回顶部 | `js/phase2.js` + `_includes/footer.html` `#back-to-top` | 滚动 >300px 显示 |
| 代码复制 | `js/phase2.js` | Clipboard API + fallback |
| 图片灯箱 | `js/phase2.js` | 点击放大，ESC 关闭 |
| 导航栏滚动 | `js/phase2.js` `initNavScroll()` | 添加 `.is-scrolled` 类 |
| 卡片渐入动画 | `js/phase2.js` `initScrollReveal()` | IntersectionObserver |
| 侧边目录 | `_includes/footer.html` 底部脚本 | jQuery One Page Nav |
| MathJax | `_includes/head.html` | **按需加载**：需 `mathjax: true` |

---

## 2. 文件结构

```
whiteicey.github.io/
├── _config.yml              # Jekyll 配置（URL、分页、评论等）
├── _includes/
│   ├── head.html            # <head>：meta、CSS、字体、CSS Variables、MathJax
│   ├── nav.html             # 导航栏 + 暗色模式切换按钮
│   └── footer.html          # 页脚 + 所有 JS 脚本加载 + 侧边目录逻辑
├── _layouts/
│   ├── default.html         # 基础布局（进度条 + nav + content + footer）
│   ├── page.html            # 通用页面布局（含侧边栏）
│   ├── post.html            # 文章布局（hero header + 文章内容 + 评论）
│   └── keynote.html         # Keynote 布局
├── _posts/                  # 博客文章（Markdown）
├── css/
│   ├── hux-blog.css         # 主样式（~2030 行，含所有定制样式）
│   ├── hux-blog.min.css     # ⚠️ 未使用（见注意事项）
│   ├── bootstrap.min.css    # Bootstrap 3
│   └── syntax.css           # 代码语法高亮
├── js/
│   ├── phase2.js            # 自定义交互（暗色模式、进度条、灯箱等）
│   ├── hux-blog.js          # 原主题 JS（导航栏滚动行为）
│   ├── hux-blog.min.js      # ⚠️ 未使用
│   ├── jquery.min.js        # jQuery
│   ├── bootstrap.min.js     # Bootstrap JS
│   ├── jquery.nav.js        # 侧边目录滚动跟踪
│   └── jquery.tagcloud.js   # 标签云
├── img/                     # 图片资源（268 张）
├── pwa/
│   ├── manifest.json        # PWA 配置
│   └── icons/               # PWA 图标
├── sw.js                    # Service Worker
├── feed.xml                 # RSS 源
├── robots.txt               # 爬虫指令
├── 404.html                 # 自定义 404 页面
├── index.html               # 首页模板
├── about.html               # 关于页
├── tags.html                # 标签页
└── package.json             # NPM 配置（Grunt 构建）
```

### ⚠️ 重要注意：CSS/JS 文件引用

`_includes/footer.html` 中引用的是**非压缩版**文件：

```html
<script src="/js/hux-blog.js"></script>    <!-- 不是 hux-blog.min.js -->
```

`_includes/head.html` 中引用的是：

```html
<link rel="stylesheet" href="/css/hux-blog.min.css">  <!-- 但实际样式在 hux-blog.css -->
```

**原因**：`.min.css` 文件是 Grunt 编译产物，但本次重设计的所有样式都添加在 `hux-blog.css` 中。如果使用 `.min.css`，需要重新运行 `grunt` 编译。当前 footer.html 已切换到非压缩版 JS，但 head.html 的 CSS 引用需确认（详见第5节排查经验）。

---

## 3. 设计系统

### 配色方案

```css
/* 亮色模式 */
--color-brand-primary: #006D77;    /* 深青色 — 链接、强调 */
--color-brand-accent:  #D4A574;    /* 暖金色 — 装饰、悬停 */
--color-text-primary:  #2D2D2D;    /* 正文 */
--color-text-secondary:#6B6B6B;    /* 辅助文字 */
--color-text-muted:    #999999;    /* 弱化文字 */
--color-bg-primary:    #F3F4F6;    /* 页面背景 */
--color-bg-card:       #FFFFFF;    /* 卡片背景 */

/* 暗色模式 */
--color-brand-primary: #4DB6AC;    /* 亮青色 */
--color-brand-accent:  #E6C9A0;    /* 亮金色 */
--color-bg-primary:    #1A1A2E;    /* 深蓝黑 */
--color-bg-card:       #16213E;    /* 深蓝灰 */
```

### 字体层次

| 用途 | 字体 | 场景 |
|------|------|------|
| UI 元素 | Inter | 导航栏、标签胶囊、页脚、h3/h5 |
| 标题/引用 | Source Serif 4 | 文章 h1/h2、hero 标题、blockquote、日期 |
| 代码 | JetBrains Mono | 行内代码、代码块 |

### 间距系统（8pt 网格）

定义在 `less/variables.less`：
- `@space-xs`: 4px / `@space-sm`: 8px / `@space-md`: 16px
- `@space-lg`: 24px / `@space-xl`: 32px / `@space-2xl`: 48px

### 阴影系统

```css
--shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
--shadow-md: 0 4px 16px rgba(0,0,0,0.08);
--shadow-lg: 0 12px 32px rgba(0,0,0,0.12);
--card-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
--card-shadow-hover: 0 8px 24px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08);
```

---

## 4. 更新记录

基准点：`backup-before-redesign` 分支（commit `298cac3`）

### Commit 1: `8fb4157` — 前端重设计主体

**改动范围：** 8 文件，+674 -105 行

| 文件 | 改动 |
|------|------|
| `index.html` | 首页重构：首篇→全宽 Featured 卡片 + 其余→CSS Grid 卡片网格 + 新分页器 |
| `_includes/head.html` | 扩展 CSS 变量（阴影/圆角/间距）+ 修复 `rel="holo-touch-icon"`→`apple-touch-icon` + 移除重复 `</script>` + 添加 noscript fallback |
| `_includes/footer.html` | 重构为三栏 grid 页脚（品牌/导航/社交）+ 移除旧 Bootstrap row/col 结构 |
| `_layouts/post.html` | 添加文章元信息条（后被移除，见 Commit 4） |
| `css/hux-blog.css` | +500 行：卡片样式、标签胶囊、排版韵律、引用块、表格、代码块、页脚、侧边目录、动画、`prefers-reduced-motion` |
| `css/syntax.css` | 修复无效 `//` CSS 注释→`/* */` |
| `js/phase2.js` | +38 行：`initNavScroll()` + `initScrollReveal()` |
| `tags.html` | 标签云→胶囊标签（`.tag-pill` + 计数） |

### Commit 2: `78d1425` — 修复三个视觉问题

| 问题 | 修复 |
|------|------|
| 导航栏毛玻璃效果太突兀 | 移除全局 `backdrop-filter`，降低 blur 12→6px，提高不透明度 0.85→0.95 |
| 文章元信息条横亘整页 | 移除全宽背景色和底部边线 |
| Catalog 滚动后变成白条 | `height: 100%` → `max-height: calc(100vh-80px)`，移除不透明背景 |

### Commit 3: `565bdd6` — 合并元信息条到 hero header

- **移除**独立的 `post-meta-bar` 组件
- 将阅读时间融入 hero header 的 `.meta` 行：`Posted by X on Date · N min read`
- `.meta` 改为 flex 布局

### Commit 4: `616c3c0` — 封面图片 + SEO 修复

| 修复 | 说明 |
|------|------|
| 分页封面图片失效 | `index.html` 图片路径添加 `{{ site.baseurl }}/` 前缀 |
| URL HTTP→HTTPS | `_config.yml` |
| 启用 RSS | `RSS: false` → `true` |
| Open Graph 标签 | `_includes/head.html` 添加 og:title/description/type/url/image |
| robots.txt | 新建，指向 sitemap |
| PWA manifest | 名称 "BY Blog" → "秋的博客 \| AUTUMN Blog" |
| Service Worker | 域名白名单 `huangxuan.me` → `whiteicey.github.io` |

### Commit 5: `d04f4b7` — 视觉精修 + 无障碍 + 性能

| 类别 | 内容 |
|------|------|
| 字体层次 | Source Serif 4 用于标题/引用，Inter 用于 UI |
| 标题装饰 | h2 左侧 gold 色边线，h5 底部 gold 装饰线 |
| 链接动画 | 悬停下划线 teal→gold 渐变 |
| 图片 | 圆角 + 阴影 |
| 进度条 | teal→gold 渐变色 |
| 导航栏 | 桌面端链接下划线滑入动画 |
| 404 页面 | 渐变 "404" + 浮动动画 + 毛玻璃返回按钮 |
| MathJax | 按需加载（`{% if page.mathjax %}`） |
| 无障碍 | focus-visible 状态 + 移动端 48px 触摸目标 |
| package.json | 更新为 whiteicey 信息 |

### Commit 6: `f5bc76a` — 修复 mathjax front matter 位置

`mathjax: true` 从 `tags:` 后面（被当作标签）移到 `tags:` 前面（正确的 front matter 字段）。

---

## 5. 已知问题与排查经验

### 问题 1：`.min.css` / `.min.js` 与源文件不同步

**症状**：修改了 `hux-blog.css` 但页面样式没变化。

**原因**：`_includes/head.html` 引用的是 `hux-blog.min.css`，但 `.min.css` 是 Grunt 编译产物，没有随源文件一起更新。

**解决**：
- 方案 A：在 `head.html` 中将 `hux-blog.min.css` 改为 `hux-blog.css`
- 方案 B：运行 `grunt` 重新编译

**经验**：修改 CSS/JS 后，始终检查 `head.html` 和 `footer.html` 中实际引用的是哪个文件。

### 问题 2：分页页面图片 404

**症状**：首页第一页图片正常，点击 "Older Posts" 后封面图消失。

**原因**：文章 front matter 中 `header-img: img/xxx.jpg`（相对路径）。在 `/page2/` 下，浏览器解析为 `/page2/img/xxx.jpg`。

**解决**：在 `index.html` 模板中使用 `{{ site.baseurl }}/{{ post.header-img }}`。

**经验**：所有模板中引用图片路径时，**必须**加上 `{{ site.baseurl }}/` 前缀。

### 问题 3：`</div>` 闭合标签与 Liquid 循环不匹配

**症状**：只有 1 篇文章或 0 篇文章时页面布局错乱。

**原因**：`<div class="post-grid">` 在 `{% if forloop.first %}` 内打开，但 `</div>` 在循环外无条件闭合。

**解决**：
- `<div class="post-grid">` 只在第 2 篇文章时打开（`{% if forloop.index == 2 %}`）
- `</div>` 用 `{% if paginator.posts.size > 1 %}` 包裹

**经验**：Liquid 模板中的 HTML 标签开闭必须考虑所有边界情况（0 篇、1 篇、多篇）。

### 问题 4：CSS 规则重复导致属性互相覆盖

**症状**：修改了引用块样式但效果不对。

**原因**：`blockquote` 在 CSS 中出现两次（原始规则 + 新增规则），两者设置不同的 padding/margin/background。

**解决**：合并为单一规则，新增的装饰元素（`::before`）单独一条规则。

**经验**：新增 CSS 前，先用 `grep -n ".selector" css/hux-blog.css` 检查是否已有同名规则。

### 问题 5：side-catalog fixed 后白条延伸到底部

**症状**：滚动到文章中间时，右侧目录变成一条白色长条延伸到页面底部。

**原因**：`.side-catalog` 设置了 `height: 100%`，在 `position: fixed` 时占满整个视口高度，加上新增的白色背景形成白条。

**解决**：`height: 100%` → `max-height: calc(100vh - 80px)`，移除不透明背景。

**经验**：`position: fixed` 元素不要使用 `height: 100%`，改用 `max-height: calc(100vh - Npx)`。

### 问题 6：`mathjax: true` 放在 `tags:` 后面

**症状**：文章中数学公式不渲染。

**原因**：`sed` 命令将 `mathjax: true` 插入到 `tags:` 下一行，Jekyll 将其解析为标签项而非 front matter 字段。

**正确格式**：
```yaml
---
catalog: true
mathjax: true      # ← 必须在 tags: 之前
tags:
    - Python
---
```

**经验**：修改 front matter 时手动验证 YAML 结构，特别是列表字段（`tags:`）周围的插入。

### 问题 7：导航栏毛玻璃效果过于突兀

**症状**：滚动页面后导航栏的 `backdrop-filter: blur(12px)` 效果过强，与页面整体风格不协调。

**解决**：降低 blur 值（12→6px），提高背景不透明度（0.85→0.95），让效果更微妙。

**经验**：`backdrop-filter` 在不同背景上效果差异很大，需要在实际页面上调试而非凭参数猜测。

---

## 6. 维护指南

### 发布新文章

```yaml
---
layout: post
title: "文章标题"
subtitle: "副标题"
date: 2026-06-01        # 使用 YYYY-MM-DD 格式
author: whiteicey
header-img: img/xxx.jpg  # 相对路径，不要加前导 /
catalog: true            # 启用侧边目录
mathjax: true            # 仅数学公式文章需要
tags:
    - 标签1
    - 标签2
---

正文内容...
```

### 修改样式

1. 编辑 `css/hux-blog.css`（**不是** `.min.css`）
2. 确认 `_includes/head.html` 引用的是正确的 CSS 文件
3. 所有颜色使用 CSS 变量（`var(--color-xxx)`），确保暗色模式兼容
4. 新增选择器前先 `grep` 检查是否已存在
5. 提交后 `git push origin main` 即可自动部署

### 修改 JavaScript

1. 编辑 `js/phase2.js`（自定义交互）或 `js/hux-blog.js`（主题原有逻辑）
2. 确认 `_includes/footer.html` 引用的是非压缩版

### 安全回滚

```bash
# 回滚到重设计前的状态
git checkout main
git reset --hard backup-before-redesign
git push origin main --force

# 回滚到特定提交
git reset --hard <commit-hash>
git push origin main --force
```

### CSS 变量新增流程

1. 在 `_includes/head.html` 的 `:root` 中定义亮色值
2. 在 `[data-theme="dark"]` 中定义暗色值
3. 在 `css/hux-blog.css` 中使用 `var(--变量名, 默认值)`

### 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 生产分支，GitHub Pages 自动部署 |
| `backup-before-redesign` | 重设计前的完整备份（commit `298cac3`） |
| `frontend-redesign` | 重设计工作分支（已合并到 main） |

---

*本文档由 Claude Code 辅助生成，基于实际开发过程整理。*
