 // tag.js
document.addEventListener('DOMContentLoaded', async function() {
    try {
      // 1. 从URL参数获取标签
      const urlParams = new URLSearchParams(window.location.search);
      const tag = urlParams.get('tag');
      
      if (!tag) {
        document.getElementById('tag-title').textContent = '未指定标签';
        return;
      }
      
      // 2. 更新标题
      document.getElementById('tag-title').textContent = `Tag: ${tag}`;
      
      // 3. 加载文章数据
      const postsData = await fetchPostsData();
      
      // 4. 过滤包含该标签的文章
      const taggedPosts = postsData.filter(post => 
        post.tags && post.tags.includes(tag)
      );
      
      // 5. 渲染结果
      const tagPostsContainer = document.getElementById('tag-posts');
      
      if (taggedPosts.length === 0) {
        tagPostsContainer.innerHTML = `<p>没有找到与 "${tag}" 标签相关的文章。</p>`;
        return;
      }
      
      let htmlContent = '';
      taggedPosts.forEach(post => {
        htmlContent += generatePostCardHTML(post);
      });
      
      tagPostsContainer.innerHTML = htmlContent;
      
    } catch (error) {
      console.error('加载标签文章失败:', error);
      document.getElementById('tag-posts').innerHTML = 
        '<p>无法加载文章，请稍后重试。</p>';
    }
  });