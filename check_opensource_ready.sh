#!/bin/bash
# 项目开源准备验证脚本
# 使用方法: bash check_opensource_ready.sh

echo "========================================"
echo "项目开源准备验证"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# 1. 检查.gitignore
echo "1. 检查Git配置..."
if [ -f ".gitignore" ]; then
    echo -e "${GREEN}✓${NC} .gitignore 已创建"
    ((PASS++))
else
    echo -e "${RED}✗${NC} .gitignore 未找到"
    ((FAIL++))
fi

# 2. 检查logs目录
if [ -d "logs" ]; then
    LOG_COUNT=$(ls logs/ 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} logs/ 目录已创建 (${LOG_COUNT}个文件)"
    ((PASS++))
else
    echo -e "${RED}✗${NC} logs/ 目录未找到"
    ((FAIL++))
fi

# 3. 检查代码中的绝对路径（排除文档）
echo ""
echo "2. 检查绝对路径..."
ABSOLUTE_PATHS=$(grep -r "/root/zhaokj/test_model" \
    --include="*.py" --include="*.sh" \
    2>/dev/null | \
    grep -v "logs/" | \
    grep -v "CLEANUP_SUMMARY" | \
    grep -v "check_opensource_ready" | \
    wc -l)

if [ "$ABSOLUTE_PATHS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} 代码中无绝对路径"
    ((PASS++))
else
    echo -e "${RED}✗${NC} 发现 ${ABSOLUTE_PATHS} 处绝对路径"
    grep -rn "/root/zhaokj/test_model" \
        --include="*.py" --include="*.sh" \
        2>/dev/null | \
        grep -v "logs/" | \
        grep -v "CLEANUP_SUMMARY" | \
        grep -v "check_opensource_ready"
    ((FAIL++))
fi

# 4. 检查必要文档
echo ""
echo "3. 检查项目文档..."
if [ -f "README.md" ]; then
    echo -e "${GREEN}✓${NC} README.md 存在"
    ((PASS++))
else
    echo -e "${RED}✗${NC} README.md 未找到"
    ((FAIL++))
fi

if [ -f "CLEANUP_SUMMARY.md" ]; then
    echo -e "${GREEN}✓${NC} CLEANUP_SUMMARY.md 存在"
    ((PASS++))
else
    echo -e "${YELLOW}!${NC} CLEANUP_SUMMARY.md 未找到（可选）"
fi

# 5. 检查大文件
echo ""
echo "4. 检查大文件..."
LARGE_FILES=$(find . -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l)
if [ "$LARGE_FILES" -gt 0 ]; then
    echo -e "${YELLOW}!${NC} 发现 ${LARGE_FILES} 个.pkl文件（请确保在.gitignore中）"
    find . -maxdepth 1 -name "*.pkl" -exec ls -lh {} \; | awk '{print "  - " $9 " (" $5 ")"}'
else
    echo -e "${GREEN}✓${NC} 无大型数据文件（或已在.gitignore）"
    ((PASS++))
fi

# 6. 检查环境配置说明
echo ""
echo "5. 检查环境配置..."
CONDA_ACTIVATE=$(grep -r "miniconda3/bin/activate" --include="*.sh" 2>/dev/null | wc -l)
if [ "$CONDA_ACTIVATE" -gt 0 ]; then
    echo -e "${YELLOW}!${NC} 发现 ${CONDA_ACTIVATE} 处conda环境激活"
    echo "  提示: 建议在README中说明用户需要根据自己环境修改"
fi

# 总结
echo ""
echo "========================================"
echo "验证结果总结"
echo "========================================"
echo -e "通过: ${GREEN}${PASS}${NC}"
echo -e "失败: ${RED}${FAIL}${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}✓ 项目已准备就绪，可以开源上传！${NC}"
    exit 0
else
    echo -e "${RED}✗ 还有 ${FAIL} 项需要处理${NC}"
    exit 1
fi
