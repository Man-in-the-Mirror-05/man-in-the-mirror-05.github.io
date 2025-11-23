# 必应每日壁纸功能说明

本项目已集成必应每日壁纸功能，会自动下载必应壁纸并设置为网站全局背景。

## 功能特性

1. **自动下载壁纸**
   - 每天自动下载zh-cn地区的必应每日壁纸
   - 保存原英文名称（如：`ThailandNewYears.jpg`）
   - 同时保存为 `latest.jpg` 供网站使用

2. **全局背景**
   - 所有页面使用 `latest.jpg` 作为背景图片
   - 浅色模式：白色半透明遮罩
   - 深色模式：黑色半透明遮罩
   - 背景图片固定，不随页面滚动

3. **图片名称显示**
   - 在网页右下角显示当前壁纸的英文名称
   - 自动适配深浅色模式
   - 移动端自适应

4. **自动同步**
   - 下载后自动提交到Git并推送到GitHub
   - 保留最近30天的元数据记录

## 文件结构

```
.
├── fetch_bing_wallpaper.py          # Python下载脚本
├── run_fetch_bing_wallpaper.bat      # Windows批处理包装
├── bing_wallpaper_metadata.json     # 元数据文件（根目录）
├── static/
│   ├── img/                         # 图片存储目录
│   │   ├── latest.jpg               # 最新壁纸（网站使用）
│   │   └── *.jpg                    # 历史壁纸（原英文名）
│   ├── css/
│   │   └── custom-background.css    # 自定义背景样式
│   └── bing_wallpaper_metadata.json # 元数据文件（网站访问）
└── layouts/
    └── partials/
        ├── extend_head.html         # 添加CSS引用
        └── extend_footer.html       # 添加图片名称显示

```

## 使用方法


### 1. 手动运行测试

```bash
python fetch_bing_wallpaper.py
```

或使用批处理文件（Windows）：
```cmd
run_fetch_bing_wallpaper.bat
```

### 2. 设置Windows定时任务

详细步骤请参考 `WINDOWS_TASK_SETUP.md` 文件。

**快速设置（PowerShell管理员模式）：**

```powershell
$action = New-ScheduledTaskAction -Execute "E:\man-in-the-mirror-05.github.io\run_fetch_bing_wallpaper.bat" -WorkingDirectory "E:\man-in-the-mirror-05.github.io"
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "Fetch Bing Wallpaper" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "每天自动下载必应每日壁纸"
```

**注意**：请将路径替换为你的实际项目路径。

## 配置说明

### 修改透明度

编辑 `static/css/custom-background.css`：

- 浅色模式透明度：修改 `rgba(255, 255, 255, 0.65)` 中的 `0.65`（0-1之间）
- 深色模式透明度：修改 `rgba(0, 0, 0, 0.65)` 中的 `0.65`（0-1之间）

### 修改下载地区

编辑 `fetch_bing_wallpaper.py`，修改 `MARKET` 变量：

```python
MARKET = "zh-cn"  # 可选：en-us, en-gb, ja-jp, fr-fr, de-de 等
```

### 修改定时任务时间

在Windows任务计划程序中修改触发时间，或使用PowerShell：

```powershell
$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"  # 改为凌晨2点
```

## 故障排除

### 1. 脚本执行失败

- 检查Python环境：`python --version`
- 检查依赖：`pip list | findstr requests`
- 检查网络连接
- 查看错误日志：`fetch_error.log`

### 2. Git推送失败

- 检查Git配置：`git config --list`
- 检查GitHub认证（SSH密钥或Personal Access Token）
- 手动测试：`git push`

### 3. 背景图片不显示

- 检查 `static/img/latest.jpg` 是否存在
- 检查CSS文件路径是否正确
- 清除浏览器缓存
- 检查Hugo构建输出

### 4. 图片名称不显示

- 检查 `static/bing_wallpaper_metadata.json` 是否存在
- 打开浏览器开发者工具查看控制台错误
- 检查JSON文件格式是否正确

## 技术细节

- **API端点**：`https://www.bing.com/HPImageArchive.aspx`
- **图片分辨率**：1920x1080
- **元数据格式**：JSON，包含日期、文件名、标题、版权信息等
- **Git提交**：自动提交到仓库，提交信息包含图片名称

## 参考

- 原文章：https://blog.tauyoung.top/article/Bing-Image-Archive/
- 必应图片API文档：https://www.bing.com/HPImageArchive.aspx

