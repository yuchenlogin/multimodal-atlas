// 主题切换（持久化）
(function(){
  var saved = localStorage.getItem('atlas-theme');
  if (!saved) saved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', saved);
  var btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', function(){
    var cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', cur);
    localStorage.setItem('atlas-theme', cur);
  });
})();
// 阅读进度
(function(){
  var bar = document.getElementById('reading-bar');
  var btn = document.getElementById('reading-progress-btn');
  if (!bar) return;
  function upd(){
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? Math.min(1, h.scrollTop / max) : 0;
    bar.style.width = (p*100).toFixed(1) + '%';
    if (btn) btn.textContent = Math.round(p*100) + '%';
  }
  document.addEventListener('scroll', upd, {passive:true}); upd();
})();
// 滚动时 MathJax 公式高度适配（无需处理，tex-svg 自适应）
