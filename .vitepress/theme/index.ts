import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './gpu.css'

const components = import.meta.glob('./components/*.vue', { eager: true }) as Record<
  string,
  { default: object }
>

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    for (const [path, mod] of Object.entries(components)) {
      const name = path.split('/').pop()!.replace('.vue', '')
      app.component(name, mod.default)
    }
  }
} satisfies Theme
