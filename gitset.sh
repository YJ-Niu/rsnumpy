#!/bin/bash
# ============================================================
# 脚本名称: git-setup-push.sh
# 功能: 设置本地项目的 Git 远程仓库并推送到指定分支
#       支持首次推送和后续更新，确保 IDE Git 正确识别
# 用法: 
#   ./gitset.sh [仓库URL] [分支名] [--force] [--help]
#   在项目目录下运行
# ============================================================

set -e  # 遇到错误立即退出

# ---------- 默认配置 ----------
DEFAULT_REPO_URL="https://gitee.com/love_develop/rsnum.git"
DEFAULT_BRANCH="master"
REMOTE_NAME="origin"
# -----------------------------

usage() {
    echo "用法: $0 [仓库URL] [分支名] [--force] [--help]"
    echo ""
    echo "参数:"
    echo "  仓库URL    远程仓库地址，如 https://gitee.com/love_develop/rsnum.git"
    echo "  分支名     目标分支，如 master 或 main"
    echo "  --force    强制推送（覆盖远程，谨慎使用）"
    echo "  --help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 https://gitee.com/love_develop/rsnum.git master"
    echo "  $0 --help"
    echo ""
    echo "注意:"
    echo "  - 此脚本需在项目根目录下运行"
    echo "  - 首次推送前请确保已添加所有需要的文件"
    exit 0
}

# 检查 git
if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git 命令，请先安装 Git"
    exit 1
fi

# 解析参数
FORCE_PUSH=false
REPO_URL="$DEFAULT_REPO_URL"
BRANCH="$DEFAULT_BRANCH"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            usage
            ;;
        --force)
            FORCE_PUSH=true
            shift
            ;;
        *)
            if [ -z "$REPO_URL" ] || [ "$REPO_URL" = "$DEFAULT_REPO_URL" ]; then
                REPO_URL="$1"
            elif [ -z "$BRANCH" ] || [ "$BRANCH" = "$DEFAULT_BRANCH" ]; then
                BRANCH="$1"
            fi
            shift
            ;;
    esac
done

# 检查是否在 git 仓库中
if [ ! -d ".git" ]; then
    echo "检测到当前目录不是 Git 仓库，正在初始化..."
    git init
    echo "✓ Git 仓库初始化完成"
fi

# 确保本地分支存在
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
if [ -z "$CURRENT_BRANCH" ]; then
    echo "当前没有活跃分支，创建并切换到 $BRANCH..."
    git checkout -b "$BRANCH"
else
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        echo "当前分支: $CURRENT_BRANCH，切换到目标分支: $BRANCH..."
        if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
            git checkout "$BRANCH"
        else
            git checkout -b "$BRANCH"
        fi
    fi
fi

echo "=========================================="
echo "本地目录: $(pwd)"
echo "目标仓库: $REPO_URL"
echo "目标分支: $BRANCH"
echo "远程名称: $REMOTE_NAME"
echo "强制推送: $FORCE_PUSH"
echo "=========================================="

# ---------- 配置远程仓库 ----------
echo ""
echo "---------- 配置远程仓库 ----------"

# 检查远程是否已存在
if git remote get-url "$REMOTE_NAME" &> /dev/null; then
    EXISTING_URL=$(git remote get-url "$REMOTE_NAME")
    if [ "$EXISTING_URL" = "$REPO_URL" ]; then
        echo "✓ 远程仓库 $REMOTE_NAME 已配置为: $REPO_URL"
    else
        echo "更新远程仓库 $REMOTE_NAME 的 URL..."
        echo "  原地址: $EXISTING_URL"
        echo "  新地址: $REPO_URL"
        git remote set-url "$REMOTE_NAME" "$REPO_URL"
        echo "✓ 远程仓库 URL 更新完成"
    fi
else
    echo "添加远程仓库 $REMOTE_NAME..."
    git remote add "$REMOTE_NAME" "$REPO_URL"
    echo "✓ 远程仓库添加完成"
fi

# ---------- 准备提交 ----------
echo ""
echo "---------- 准备提交 ----------"

# 检查是否有未提交的文件
if ! git diff --quiet --cached --exit-code && ! git diff --quiet --exit-code; then
    echo "检测到未提交的更改，添加所有文件..."
    git add -A
    echo "✓ 文件已添加到暂存区"
    
    # 检查是否有提交历史
    if ! git rev-parse --verify HEAD &> /dev/null; then
        echo "无提交历史，创建初始提交..."
        git commit -m "Initial commit"
        echo "✓ 初始提交创建完成"
    else
        echo "检测到已有提交历史，跳过自动提交"
        echo "提示: 如果需要提交，请手动执行 git commit"
    fi
else
    echo "✓ 工作目录干净，无需添加文件"
fi

# 检查是否有提交
if ! git rev-parse --verify HEAD &> /dev/null; then
    echo "错误: 当前分支没有任何提交，请先添加文件并提交"
    echo "建议执行:"
    echo "  git add -A"
    echo "  git commit -m \"Initial commit\""
    exit 1
fi

# ---------- 推送代码 ----------
echo ""
echo "---------- 推送代码 ----------"

PUSH_CMD="git push"
if $FORCE_PUSH; then
    PUSH_CMD="git push --force"
    echo "⚠️  注意: 使用强制推送模式"
fi

echo "推送分支 $BRANCH 到远程仓库..."

# 首次推送需要设置 upstream
if ! git rev-parse --abbrev-ref "$BRANCH@{upstream}" &> /dev/null; then
    echo "首次推送，设置 upstream..."
    $PUSH_CMD -u "$REMOTE_NAME" "$BRANCH"
else
    echo "已有 upstream，直接推送..."
    $PUSH_CMD "$REMOTE_NAME" "$BRANCH"
fi

echo "✅ 推送成功!"

# ---------- 辅助 IDE 正确识别仓库 ----------
echo ""
echo "========== IDE 辅助设置 =========="

# 1. 检查 .git 目录是否存在（确保 IDE 能识别）
if [ -d ".git" ]; then
    echo "✓ .git 目录存在，IDE 将正确识别 Git 仓库"
else
    echo "⚠️  警告: .git 目录不存在"
fi

# 2. 显示当前仓库状态
echo ""
echo "当前仓库状态:"
echo "  远程仓库: $(git remote get-url "$REMOTE_NAME")"
echo "  当前分支: $(git branch --show-current)"
echo "  远程状态: $(git rev-parse --abbrev-ref "$BRANCH@{upstream}" 2>/dev/null || echo "未设置 upstream")"

# 3. 生成 .code-workspace 文件（可选）
WORKSPACE_FILE="$(basename "$(pwd)").code-workspace"
cat > "$WORKSPACE_FILE" <<EOF
{
    "folders": [
        {
            "path": "$(pwd)",
            "name": "$(basename "$(pwd)")"
        }
    ],
    "settings": {
        "git.enabled": true,
        "git.path": ""
    }
}
EOF
echo ""
echo "✓ 已生成 VSCode 工作区文件: $WORKSPACE_FILE"
echo "  双击此文件即可在 VSCode 中打开正确的仓库根目录"

# 4. 重要提醒
echo ""
echo "⚠️  重要提醒:"
echo "   - 请确保在 IDE 中打开的是仓库根目录: $(pwd)"
echo "   - 不要打开父目录，否则 Git 插件可能无法正确识别"
echo "   - 如果 IDE 仍无法识别，请重启 IDE 并重新打开此目录"

echo ""
echo "=========================================="
echo "🎉 操作完成!"