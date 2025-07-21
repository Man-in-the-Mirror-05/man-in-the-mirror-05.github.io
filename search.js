 // search.js
document.addEventListener('DOMContentLoaded', async function() {
    try {
      // 1. 获取DOM元素
      const searchInput = document.getElementById('search-query');
      const searchButton = document.getElementById('search-button');
      const resultsContainer = document.getElementById('search-results');
      
      // 2. 加载文章数据
      const postsData = await fetchPostsData();
      
      // 3. 初始显示所有文章
      displayResults(postsData, resultsContainer);
      
      // 4. 搜索功能
      searchButton.addEventListener('click', () => performSearch(postsData, resultsContainer));
      
      searchInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') {
          performSearch(postsData, resultsContainer);
        }
      });
      
    } catch (error) {
      console.error('搜索功能初始化失败:', error);
    }
  });
  
  function performSearch(postsData, container) {
    const query = document.getElementById('search-query').value.trim().toLowerCase();
    
    if (query === '') {
      // 如果搜索框为空，显示所有文章
      displayResults(postsData, container);
      return;
    }
    
    // 执行搜索
    const filteredPosts = postsData.filter(post => {
      // 匹配标题、摘要、标签
      return post.title.toLowerCase().includes(query) ||
             post.excerpt.toLowerCase().includes(query) ||
             (post.tags && post.tags.some(tag => tag.toLowerCase().includes(query)));
    });
    
    displayResults(filteredPosts, container);
  }
  
  function displayResults(posts, container) {
    if (posts.length === 0) {
      container.innerHTML = '<p>没有找到匹配的文章。</p>';
      return;
    }
    
    let htmlContent = '';
    posts.forEach(post => {
      htmlContent += generatePostCardHTML(post);
    });
    
    container.innerHTML = htmlContent;
  }