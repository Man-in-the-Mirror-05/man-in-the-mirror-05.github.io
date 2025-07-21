// 公共函数：获取文章数据
async function fetchPostsData() {
  try {
    const response = await fetch('posts.json');
    const postsData = await response.json();
    return postsData;
  } catch (error) {
    console.error('加载文章数据失败:', error);
    return [];
  }
}

// 公共函数：生成文章卡片HTML
function generatePostCardHTML(post) {
  return `
    <article class="card">
      <div class="card-content">
        <h3 class="card-title">${post.title}</h3>
        <p class="card-excerpt">${post.excerpt}</p>
        <div class="card-meta">
          <span>Date: ${post.date}</span>
          <span>Estimated Reading Time: ${post.readingTime}</span>
          <span>Author: ${post.author}</span>
        </div>
      </div>
    </article>
  `;
}