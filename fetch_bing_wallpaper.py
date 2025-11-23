#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
必应每日壁纸下载脚本
每天自动下载zh-cn地区的必应壁纸，保存原英文名，复制为latest，并同步到GitHub
"""

import os
import sys
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import requests

# 配置
BING_API_URL = "https://www.bing.com/HPImageArchive.aspx"
MARKET_EN = "en-us"  # 优先使用英语版本获取文本信息
MARKET_CN = "zh-cn"  # 用于下载图片
IMG_DIR = Path("static/img")
METADATA_FILE = "bing_wallpaper_metadata.json"

def ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)

def fetch_bing_image_metadata(market):
    """获取必应图片元数据"""
    params = {
        "format": "xml",
        "idx": 0,
        "n": 1,
        "mkt": market
    }
    
    try:
        response = requests.get(BING_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"获取必应图片元数据失败 (market={market}): {e}")
        return None

def parse_xml_metadata(xml_content):
    """解析XML元数据"""
    try:
        root = ET.fromstring(xml_content)
        image = root.find("image")
        if image is None:
            raise ValueError("未找到图片数据")
        
        enddate = image.find("enddate").text
        url_base = image.find("urlBase").text
        headline = image.find("headline").text if image.find("headline") is not None else ""
        copyright = image.find("copyright").text if image.find("copyright") is not None else ""
        
        # 从urlBase提取文件名（例如：/th?id=OHR.ThailandNewYears_ZH-CN2058192262）
        # 提取ThailandNewYears部分
        if "OHR." in url_base:
            filename = url_base.split("OHR.")[1].split("_")[0]
        else:
            # 如果没有OHR.前缀，尝试其他方式提取
            filename = url_base.split("/")[-1].split("_")[0]
        
        return {
            "enddate": enddate,
            "urlBase": url_base,
            "filename": filename,
            "headline": headline,
            "copyright": copyright
        }
    except ET.ParseError as e:
        print(f"解析XML失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理元数据失败: {e}")
        sys.exit(1)

def download_image(url_base, filename):
    """下载图片"""
    # 构建完整URL（1920x1080分辨率）
    image_url = f"https://www.bing.com{url_base}_1920x1080.jpg"
    
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # 保存原英文名文件
        original_path = IMG_DIR / f"{filename}.jpg"
        with open(original_path, "wb") as f:
            f.write(response.content)
        
        # 复制为latest.jpg
        latest_path = IMG_DIR / "latest.jpg"
        shutil.copy2(original_path, latest_path)
        
        print(f"图片已下载: {original_path}")
        print(f"已复制为: {latest_path}")
        return True
    except requests.RequestException as e:
        print(f"下载图片失败: {e}")
        return False

def save_metadata(metadata):
    """保存元数据到JSON文件"""
    metadata_path = Path(METADATA_FILE)
    static_metadata_path = Path("static") / METADATA_FILE
    
    # 读取现有元数据（如果存在）
    all_metadata = []
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
        except:
            pass
    
    # 检查是否已有同一天的记录
    today_enddate = metadata["enddate"]
    # 移除同一天的所有记录（可能有多条：中文和英文）
    all_metadata = [item for item in all_metadata if item.get("enddate") != today_enddate]
    
    # 添加新元数据（优先保存英语版本）
    metadata["download_time"] = datetime.now().isoformat()
    all_metadata.append(metadata)
    
    # 只保留最近30天的记录
    all_metadata = all_metadata[-30:]
    
    # 保存到根目录和static目录
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    with open(static_metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

def get_latest_metadata():
    """获取最新的元数据"""
    metadata_path = Path(METADATA_FILE)
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
                if all_metadata:
                    return all_metadata[-1]
        except:
            pass
    return None

def git_commit_and_push():
    """提交并推送到GitHub"""
    try:
        # 获取最新图片日期
        latest_meta = get_latest_metadata()
        commit_msg = f"Fetch Bing wallpaper: {latest_meta['filename'] if latest_meta else datetime.now().strftime('%Y%m%d')}"
        
        # 添加文件（只添加壁纸相关的文件）
        files_to_add = [
            "static/img/",
            "static/" + METADATA_FILE,
            METADATA_FILE
        ]
        
        # 检查每个文件是否存在且有更改
        has_changes = False
        for file_path in files_to_add:
            result = subprocess.run(
                ["git", "status", "--porcelain", file_path],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                has_changes = True
                break
        
        if not has_changes:
            print("没有壁纸相关的更改需要提交")
            return
        
        # 添加文件
        subprocess.run(
            ["git", "add"] + files_to_add,
            check=True,
            capture_output=True
        )
        
        # 检查暂存区是否有内容
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            print("没有更改需要提交（文件可能已被提交）")
            return
        
        # 提交
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"已提交: {commit_msg}")
        
        # 推送
        subprocess.run(
            ["git", "push"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("已成功推送到GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Git操作失败: {e}")
        if e.stderr:
            print(f"错误详情: {e.stderr}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        print("请手动提交和推送更改")
    except FileNotFoundError:
        print("未找到git命令，请确保已安装Git")
        print("请手动提交和推送更改")
    except Exception as e:
        print(f"Git操作出错: {e}")
        print("请手动提交和推送更改")

def main():
    """主函数"""
    print("开始获取必应每日壁纸...")
    
    # 确保目录存在
    ensure_dir(IMG_DIR)
    
    # 优先获取英语版本的元数据（用于headline, filename, copyright）
    print("正在获取英语版本的图片元数据...")
    xml_content_en = fetch_bing_image_metadata(MARKET_EN)
    metadata_en = None
    if xml_content_en:
        try:
            metadata_en = parse_xml_metadata(xml_content_en)
            print(f"英语版本获取成功")
        except Exception as e:
            print(f"解析英语版本元数据失败: {e}")
    
    # 获取中文版本的元数据（用于urlBase和下载图片）
    print("正在获取中文版本的图片元数据...")
    xml_content_cn = fetch_bing_image_metadata(MARKET_CN)
    if not xml_content_cn:
        print("无法获取中文版本元数据，退出")
        sys.exit(1)
    
    metadata_cn = parse_xml_metadata(xml_content_cn)
    
    # 合并元数据：优先使用英语版本的文本信息，使用中文版本的urlBase
    if metadata_en:
        metadata = {
            "enddate": metadata_cn["enddate"],  # 使用中文版本的日期
            "urlBase": metadata_cn["urlBase"],   # 使用中文版本的URL（用于下载）
            "filename": metadata_en["filename"],  # 优先使用英语版本的文件名
            "headline": metadata_en["headline"],  # 优先使用英语版本的标题
            "copyright": metadata_en["copyright"]  # 优先使用英语版本的版权信息
        }
    else:
        # 如果英语版本获取失败，使用中文版本
        print("警告：英语版本获取失败，使用中文版本")
        print("建议检查网络连接或代理设置")
        metadata = metadata_cn
    
    print(f"图片名称: {metadata['filename']}")
    print(f"发布日期: {metadata['enddate']}")
    
    # 检查文件是否已存在
    original_path = IMG_DIR / f"{metadata['filename']}.jpg"
    if original_path.exists():
        print(f"图片已存在: {original_path}")
        # 即使已存在，也更新latest.jpg
        latest_path = IMG_DIR / "latest.jpg"
        shutil.copy2(original_path, latest_path)
        print(f"已更新: {latest_path}")
    else:
        # 下载图片（使用中文版本的urlBase）
        print("正在下载图片...")
        if download_image(metadata_cn['urlBase'], metadata['filename']):
            # 保存元数据
            save_metadata(metadata)
            print("元数据已保存")
        else:
            sys.exit(1)
    
    # 同步到GitHub
    print("正在同步到GitHub...")
    git_commit_and_push()
    
    print("完成！")

if __name__ == "__main__":
    main()

