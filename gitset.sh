#!/bin/bash
# ============================================================
# 脚本名称: git-sync.sh
# 功能: 自动克隆/更新 Git 仓库并切换到指定分支，
#       同时辅助 IDE（VSCode）正确识别仓库根目录。
# 用法: 
#   ./git-sync.sh [仓库URL] [分支名] [目标目录] [--no-open]
# ============================================================

set -e  # 遇到错误立即退出

# ---------- 默认配置 ----------
DEFAULT_REPO_URL="https://gitee.com/love_develop/rsnum.git"
DEFAULT_BRANCH="master"
DEFAULT_TARGET_DIR=""   # 留空则自动从仓库名提取
# -----------------------------

usage() {
    echo "用法: $0 [仓库URL] [分支名] [目标目录] [--no-open]"
    echo "示例: $0 https://gitee.com/love_develop/rsnum.git master ./my_repo"
    echo "      $0 --no-open   # 使用默认值，但不自动打开 VSCode"
    echo "若不提供参数，则使用脚本内预设的默认值"
    exit 1
}

# 检查 git
if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git 命令，请先安装 Git"
    exit 1
fi

# 解析参数，支持 --no-open
AUTO_OPEN=true
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-open)
            AUTO_OPEN=false
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

REPO_URL="${1:-$DEFAULT_REPO_URL}"
BRANCH="${2:-$DEFAULT_BRANCH}"

if [ -n "$3" ]; then
    TARGET_DIR="$3"
else
    if [ -n "$DEFAULT_TARGET_DIR" ]; then
        TARGET_DIR="$DEFAULT_TARGET_DIR"
    else
        REPO_NAME=$(basename "$REPO_URL" .git)
        TARGET_DIR="$REPO_NAME"
    fi
fi

# 转换为绝对路径
if [[ "$TARGET_DIR" = /* ]]; then
    ABS_TARGET_DIR="$TARGET_DIR"
else
    ABS_TARGET_DIR="$(pwd)/$TARGET_DIR"
fi

echo "=========================================="
echo "仓库地址: $REPO_URL"
echo "目标分支: $BRANCH"
echo "本地目录: $ABS_TARGET_DIR"
echo "=========================================="

# ---------- Git 操作 ----------
if [ -d "$TARGET_DIR" ]; then
    echo "目录已存在: $TARGET_DIR，准备更新并切换分支..."
    cd "$TARGET_DIR"
    git fetch --all --prune
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
    elif git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        git checkout -b "$BRANCH" "origin/$BRANCH"
    else
        echo "错误: 分支 $BRANCH 在本地和远程都不存在"
        exit 1
    fi
    git pull origin "$BRANCH"
    echo "更新完成，当前分支: $(git branch --show-current)"
else
    echo "克隆仓库 (单分支) 到 $TARGET_DIR ..."
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
    echo "克隆完成，当前分支: $(git branch --show-current)"
fi

echo "✅ Git 操作成功"

# ---------- 辅助 IDE 正确识别仓库 ----------
echo ""
echo "========== IDE 辅助指引 =========="

# 1. 生成 .code-workspace 文件（可选）
WORKSPACE_FILE="${ABS_TARGET_DIR}.code-workspace"
cat > "$WORKSPACE_FILE" <<EOF
{
    "folders": [
        {
            "path": "$ABS_TARGET_DIR",
            "name": "$(basename "$ABS_TARGET_DIR")"
        }
    ],
    "settings": {
        "git.enabled": true
    }
}
EOF
echo "✓ 已生成 VSCode 工作区文件: $WORKSPACE_FILE"
echo "  双击此文件即可在 VSCode 中打开正确的仓库根目录。"

# 2. 提示用户不要打开父目录
PARENT_DIR="$(dirname "$ABS_TARGET_DIR")"
echo ""
echo "⚠️  重要提醒："
echo "   - 请不要直接在 VSCode 中打开父目录: $PARENT_DIR"
echo "   - 必须打开仓库根目录: $ABS_TARGET_DIR"
echo "   - 或者打开上面生成的工作区文件: $WORKSPACE_FILE"

# 3. 自动打开 VSCode（如果允许且 code 命令存在）
if [ "$AUTO_OPEN" = true ] && command -v code &> /dev/null; then
    echo ""
    echo "检测到 VSCode 命令行工具 'code'，正在自动打开仓库..."
    code "$ABS_TARGET_DIR"
    echo "✓ VSCode 已打开，Git 插件将正确识别当前仓库。"
elif [ "$AUTO_OPEN" = true ]; then
    echo ""
    echo "未找到 'code' 命令，无法自动打开 VSCode。"
    echo "请手动使用 VSCode 打开目录: $ABS_TARGET_DIR"
else
    echo ""
    echo "已禁用自动打开（--no-open）。请手动打开目录: $ABS_TARGET_DIR"
fi

echo "=========================================="