// 文章数据加载器
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // 1. 加载文章元数据索引
        const response = await fetch('posts.json');
        const postsData = await response.json();
        
        // 2. 生成文章卡片
        const postsContainer = document.getElementById('posts');
        let htmlContent = '';
        
        for (const post of postsData) {
            htmlContent += `
                <article class="post-card">
                    <div class="post-content">
                        <h3 class="post-title">${post.title}</h3>
                        <p class="post-excerpt">${post.excerpt}</p>
                        <div class="post-meta">
                            <span>Date: ${post.date}</span>
                            <span>Estimated Reading Time: ${post.readingTime}</span>
                            <span>Author: ${post.author}</span>
                        </div>
                    </div>
                </article>
            `;
        }
        
        postsContainer.innerHTML = htmlContent;
        
        // 3. 实现搜索功能
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = '搜索文章...';
        searchInput.classList.add('search-bar');
        document.querySelector('.welcome-section').appendChild(searchInput);
        
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            document.querySelectorAll('.post-card').forEach(card => {
                const title = card.querySelector('.post-title').textContent.toLowerCase();
                const excerpt = card.querySelector('.post-excerpt').textContent.toLowerCase();
                card.style.display = (title.includes(searchTerm) || excerpt.includes(searchTerm)) 
                    ? 'block' : 'none';
            });
        });
        
    } catch (error) {
        console.error('加载文章数据失败:', error);
    }
});