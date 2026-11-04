import { defineConfig } from 'vitepress'

const operatorSidebar = [
  {
    text: 'CUDA 算子优化',
    items: [
      { text: 'Reduce 算子优化图解', link: '/reduce/reduce' },
      { text: 'GEMM 算子优化图解', link: '/gemm/gemm' },
      { text: 'Softmax 优化图解', link: '/softmax/softmax' },
      { text: 'Online Softmax 图解', link: '/softmax/online-softmax' }
    ]
  },
  {
    text: 'FlashAttention',
    items: [
      { text: 'FlashAttention V1 图解', link: '/flash-attention/v1' },
      { text: 'FlashAttention V2 图解', link: '/flash-attention/v2' }
    ]
  }
]

export default defineConfig({
  // 部署在 GitHub Pages 的项目路径下
  base: '/operator/',
  lang: 'zh-CN',
  title: 'Operator',
  description: '从硬件结构到 AI 负载，图解 GPU 基础与算子优化',
  // README、算子代码和本地的原始材料都不作为页面发布
  srcExclude: ['README.md', 'code/**', 'sources/**'],
  markdown: { math: true },
  themeConfig: {
    nav: [
      { text: 'GPU 基础', link: '/about_gpu/gpu' },
      { text: '算子优化', link: '/reduce/reduce', activeMatch: '^/(reduce|gemm|softmax|flash-attention)/' }
    ],
    sidebar: {
      '/about_gpu/': [
        {
          text: 'GPU 基础',
          items: [
            { text: 'GPU 架构图解', link: '/about_gpu/gpu' },
            { text: 'AI 计算与性能分析', link: '/about_gpu/ai-compute' },
            { text: '显存、搬移与互联', link: '/about_gpu/memory-interconnect' }
          ]
        }
      ],
      '/reduce/': operatorSidebar,
      '/gemm/': operatorSidebar,
      '/softmax/': operatorSidebar,
      '/flash-attention/': operatorSidebar
    },
    footer: {
      message: '算子优化文章以 <a href="https://github.com/caomaolufei/AIInfraGuide" target="_blank" rel="noreferrer">AI Infra Guide</a> 为蓝本重写，感谢原作者的整理。'
    },
    outline: { level: [2, 3], label: '本页目录' },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },
    docFooter: { prev: '上一页', next: '下一页' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})
