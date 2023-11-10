import traceback
import time
import logging
logger = logging.getLogger(__name__)


def with_telegram(orig_fn):
    start = time.time()
    try:
        import telegram_send

        def push_msg_fn(msg):
            now = time.time()
            diff_m = (now - start) / 60
            if diff_m > 1:
                try:
                    telegram_send.send(messages=[str(msg)])
                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                    print('push notification failed')
    except Exception:
        push_msg_fn = lambda msg: None

    def new_fn(*args, **kwargs):
        try:
            return orig_fn(*args, **kwargs)
        except Exception as e:
            push_msg_fn(e)
            raise

    return new_fn


def send_notification(msg, include_hostname=True, use_markdown=True):
    """ Send telegram notification.

    When using markdown, the following formatting can be applied:
    https://core.telegram.org/bots/api#formatting-options

    *bold \*text*
    _italic \*text_
    __underline__
    ~strikethrough~
    ||spoiler||
    *bold _italic bold ~italic bold strikethrough ||italic bold strikethrough spoiler||~ __underline italic bold___ bold*
    [inline URL](http://www.example.com/)
    [inline mention of a user](tg://user?id=123456789)
    `inline fixed-width code`
    ```
    pre-formatted fixed-width code block
    ```
    ```python
    pre-formatted fixed-width code block written in the Python programming language
    ```
    """
    try:
        import telegram_send
        try:
            import socket
            hostname = socket.gethostname()
            to_send = f'{hostname}: {msg}'
        except Exception:
            hostname = None
            to_send = str(msg)

        parse_mode = 'markdown' if use_markdown else 'text'
        telegram_send.send(messages=[to_send], parse_mode=parse_mode)
    except Exception:
        logger.exception("Could not send Telegram notification")


@with_telegram
def main():
    print('waiting for a minute')
    import time
    time.sleep(60 + 5)
    a = b + c  # should fail horribly
    return 0


if __name__ == '__main__':
    main()
