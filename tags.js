 // tags.js
document.addEventListener('DOMContentLoaded', async function() {
    try {
      // 1. 加载文章数据
      const postsData = await fetchPostsData();
      
      // 2. 提取所有标签并计数
      const tagCounts = {};
      
      postsData.forEach(post => {
        if (post.tags && Array.isArray(post.tags)) {
          post.tags.forEach(tag => {
            if (!tagCounts[tag]) {
              tagCounts[tag] = 0;
            }
            tagCounts[tag]++;
          });
        }
      });
      
      // 3. 将标签转换为数组并按出现次数排序
      const tagsArray = Object.keys(tagCounts)
        .map(tag => ({ tag, count: tagCounts[tag] }))
        .sort((a, b) => b.count - a.count);
      
      // 4. 生成标签云HTML
      const tagsContainer = document.getElementById('tags-container');
      let htmlContent = '<div class="tags-cloud">';
      
      tagsArray.forEach(tagObj => {
        htmlContent += `
          <a href="tag.html?tag=${encodeURIComponent(tagObj.tag)}" class="tag">
            ${tagObj.tag} <span class="tag-count">${tagObj.count}</span>
          </a>
        `;
      });
      
      htmlContent += '</div>';
      tagsContainer.innerHTML = htmlContent;
      
    } catch (error) {
      console.error('加载标签失败:', error);
      tagsContainer.innerHTML = '<p>无法加载标签，请稍后重试。</p>';
    }
  });