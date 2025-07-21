// archive.js
document.addEventListener('DOMContentLoaded', async function() {
    try {
      // 1. 加载文章数据
      const postsData = await fetchPostsData();
      
      // 2. 按日期倒序排序（最新在前）
      const sortedPosts = [...postsData].sort((a, b) => {
        return new Date(b.date) - new Date(a.date);
      });
      
      // 3. 按年份和月份分组
      const groupedPosts = {};
      sortedPosts.forEach(post => {
        const date = new Date(post.date);
        const year = date.getFullYear();
        const month = date.toLocaleString('default', { month: 'long' });
        
        if (!groupedPosts[year]) {
          groupedPosts[year] = {};
        }
        
        if (!groupedPosts[year][month]) {
          groupedPosts[year][month] = [];
        }
        
        groupedPosts[year][month].push(post);
      });
      
      // 4. 生成归档页面HTML
      const archiveContainer = document.getElementById('archive-container');
      let htmlContent = '';
      
      // 按年份倒序排列
      const years = Object.keys(groupedPosts).sort((a, b) => b - a);
      
      years.forEach(year => {
        htmlContent += `<div class="archive-group">`;
        htmlContent += `<h2 class="archive-year">${year}</h2>`;
        
        // 获取该年份的所有月份
        const months = Object.keys(groupedPosts[year]);
        
        // 按月份倒序排列（需要转换月份为数字排序）
        const monthMap = {
          January: 1, February: 2, March: 3, April: 4, May: 5, June: 6,
          July: 7, August: 8, September: 9, October: 10, November: 11, December: 12
        };
        
        const sortedMonths = months.sort((a, b) => monthMap[b] - monthMap[a]);
        
        sortedMonths.forEach(month => {
          htmlContent += `<h3 class="archive-month">${month}</h3>`;
          
          groupedPosts[year][month].forEach(post => {
            htmlContent += generatePostCardHTML(post);
          });
        });
        
        htmlContent += `</div>`;
      });
      
      archiveContainer.innerHTML = htmlContent;
      
    } catch (error) {
      console.error('加载归档数据失败:', error);
      archiveContainer.innerHTML = '<p>无法加载归档内容，请稍后重试。</p>';
    }
  });