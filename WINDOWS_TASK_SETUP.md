# Windows定时任务设置说明

## 设置步骤

### 方法一：使用任务计划程序GUI

1. 打开"任务计划程序"
   - 按 `Win + R`，输入 `taskschd.msc`，回车

2. 创建基本任务
   - 在右侧操作面板点击"创建基本任务"
   - 名称：`Fetch Bing Wallpaper`
   - 描述：`每天自动下载必应每日一图`

3. 设置触发器
   - 选择"每天"
   - 开始时间：设置为每天凌晨（例如：`00:00:00`）
   - 重复任务间隔：1天

4. 设置操作
   - 操作类型：选择"启动程序"
   - 程序或脚本：浏览选择 `run_fetch_bing_wallpaper.bat`
   - 起始于（可选）：设置为项目根目录路径（例如：`E:\man-in-the-mirror-05.github.io`）

5. 完成设置
   - 勾选"当单击完成时，打开此任务属性的对话框"
   - 在属性对话框中：
     - 勾选"不管用户是否登录都要运行"
     - 勾选"使用最高权限运行"
     - 在"条件"选项卡中，取消勾选"只有在计算机使用交流电源时才启动此任务"（如果需要）

### 方法二：使用PowerShell命令

以管理员身份打开PowerShell，执行以下命令：

```powershell
$action = New-ScheduledTaskAction -Execute "E:\man-in-the-mirror-05.github.io\run_fetch_bing_wallpaper.bat" -WorkingDirectory "E:\man-in-the-mirror-05.github.io"
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "Fetch Bing Wallpaper" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "每天自动下载必应每日一图"
```

**注意**：请将路径 `E:\man-in-the-mirror-05.github.io` 替换为你的实际项目路径。

## 前置要求

1. **Python环境**
   - 确保已安装Python 3.6+
   - 安装所需依赖：`pip install requests`

2. **Git配置**
   - 确保已安装Git
   - 确保Git已配置用户信息（用于提交）
   - 确保已配置GitHub远程仓库和认证（SSH密钥或Personal Access Token）

3. **网络连接**
   - 确保计算机在指定时间有网络连接
   - 如果使用代理，需要在Python脚本中配置代理设置

## 测试

手动运行测试：
```cmd
cd E:\man-in-the-mirror-05.github.io
python fetch_bing_wallpaper.py
```

或者运行批处理文件：
```cmd
run_fetch_bing_wallpaper.bat
```

## 查看日志

- 成功执行：脚本会在控制台输出信息
- 错误日志：如果使用批处理文件，错误会记录到 `fetch_error.log`

## 注意事项

1. 确保任务计划程序服务正在运行
2. 如果脚本需要访问网络代理，可能需要配置环境变量
3. 首次运行前，建议手动执行一次脚本，确保所有依赖都正确安装
4. 如果Git推送失败，需要检查GitHub认证配置

