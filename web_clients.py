from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # 避免未安装 playwright 时连纯 API 模式都无法 import
    from playwright.sync_api import Page

from page_locator import (
    PageLocator,
    PageLocatorConfig,
    ZHIPU_LOCATOR,
    WENXIN_LOCATOR,
    QWEN_LOCATOR,
    DEEPSEEK_LOCATOR,
)


class BaseWebModelClient:
    """网页模型客户端基类"""

    name: str = "unknown"
    url: str = ""
    locator_config: PageLocatorConfig | None = None

    def __init__(
        self,
        headless: bool = False,
        login_wait: bool = True,
    ):
        self.headless = headless
        self.login_wait = login_wait
        self._browser = None
        self._page = None
        self._playwright = None
        self.locator = PageLocator(self.locator_config)

    def start_session(self):
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # 给出可执行的修复提示
            raise RuntimeError(
                "网页模式需要 playwright：请先执行 "
                "`pip install playwright` 和 `playwright install chromium`"
            ) from e
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()
            self._page.goto(self.url, timeout=30000)
            if self.login_wait:
                print(f"[{self.name}] 请在浏览器中完成登录，然后按回车继续……")
                input()
            self._wait_ready(self._page)
        except BaseException:
            # 启动失败要立刻回收浏览器/驱动，否则路由器换下一个模型时会残留进程
            self.close_session()
            raise

    def close_session(self):
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._page = None
        self._playwright = None

    def _wait_ready(self, page: "Page"):
        page.wait_for_selector(
            "textarea, [contenteditable='true'], input[type='text']",
            timeout=30000,
        )

    def ask(self, prompt: str, timeout: int = 120) -> str:
        self.start_session()
        page = self._page
        # 关键：基线必须在发送前采集，否则应答页渲染太快会把新回答当成历史内容
        baseline = self.locator.snapshot(page)
        self._fill_and_send(page, prompt)
        return self.locator.find_answer(
            page, timeout=timeout, question=prompt, baseline=baseline
        )

    def _fill_and_send(self, page: "Page", text: str):
        box = self.locator.find_input(page)

        # 先聚焦，部分富文本框不聚焦不接收输入
        try:
            box.click(timeout=3000)
        except Exception:
            pass

        box.fill(text)
        page.wait_for_timeout(500)

        send = self.locator.find_send_button(page)
        sent = False
        if send is not None:
            try:
                send.click(timeout=5000)
                sent = True
            except Exception as e:
                print(f"[{self.name}] 点击发送失败（{e}），改用回车发送")

        if not sent:
            box.press("Enter")
            page.wait_for_timeout(1000)
            # 回车没生效（输入框里还留着内容）时，退一步点输入框所在容器里的最后一个按钮
            if self._read_input(box):
                clicked = False
                try:
                    clicked = box.evaluate(
                        """(el) => {
                            let root = el.closest('form')
                                || el.closest('[class*=editor], [class*=input], [class*=chat]');
                            if (!root) return false;
                            const btns = [...root.querySelectorAll('button')].filter(
                                b => b.getBoundingClientRect().width > 0 && !b.disabled);
                            if (!btns.length) return false;
                            btns[btns.length - 1].click();
                            return true;
                        }"""
                    )
                except Exception:
                    clicked = False
                if clicked:
                    page.wait_for_timeout(800)

        page.wait_for_timeout(300)

        # 清空输入框，避免下一轮把上次的问题再发一遍
        try:
            box.fill("")
        except Exception:
            pass

    @staticmethod
    def _read_input(box) -> str:
        """读取输入框当前内容（兼容 textarea 和 contenteditable）。"""
        try:
            return (box.input_value(timeout=800) or "").strip()
        except Exception:
            try:
                return (box.inner_text(timeout=800) or "").strip()
            except Exception:
                return ""


class ZhipuWebClient(BaseWebModelClient):
    name = "zhipu-web"
    url = "https://chatglm.cn/"
    locator_config = ZHIPU_LOCATOR

class DeepseekWebClient(BaseWebModelClient):
    name = "deepseek-web"
    url = "https://chat.deepseek.com/"
    locator_config = DEEPSEEK_LOCATOR


class WenxinWebClient(BaseWebModelClient):
    name = "wenxin-web"
    url = "https://yiyan.baidu.com/"
    locator_config = WENXIN_LOCATOR


class QwenWebClient(BaseWebModelClient):
    name = "qwen-web"
    url = "https://www.qianwen.com/"
    locator_config = QWEN_LOCATOR