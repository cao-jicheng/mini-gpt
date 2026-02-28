import os
import sys
import codecs
import argparse

def clean_utf8_file(input_path, output_path, chunk_size=1048576, show_progress=True):
    """
    清理输入文件中的非 UTF-8 字符，将结果写入输出文件，并显示进度。

    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param chunk_size: 读取块大小（字节），默认1MB
    :param show_progress: 是否显示进度条（默认 True）
    """
    total_size = None
    if show_progress:
        try:
            total_size = os.path.getsize(input_path)
        except OSError:
            total_size = None

    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    processed = 0
    with open(input_path, 'rb') as fin, open(output_path, 'w', encoding="utf-8", newline='') as fout:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            decoded = decoder.decode(chunk)
            fout.write(decoded)
            processed += len(chunk)

            if show_progress and (processed % (chunk_size * 10) == 0 or not chunk):
                _update_progress(processed, total_size)

        final_decoded = decoder.decode(b'', final=True)
        fout.write(final_decoded)
        processed += 0
        if show_progress:
            _update_progress(processed, total_size, final=True)

    if show_progress:
        print()
    print(f"清理完成！结果已保存至：{output_path}")

def _update_progress(processed, total_size, final=False):
    """
    在终端同一行更新进度信息。
    """
    if total_size:
        percent = processed / total_size * 100
        bar_len = 40
        filled_len = int(bar_len * processed // total_size) if total_size else 0
        bar = '█' * filled_len + '─' * (bar_len - filled_len)
        # 格式化：百分比 进度条 已处理/总大小
        sys.stdout.write(f"\r{percent:5.1f}% |{bar}| {processed}/{total_size} 字节")
    else:
        # 无法获取总大小时，只显示已处理字节数
        sys.stdout.write(f"\r已处理 {processed} 字节")

    sys.stdout.flush()
    if final:
        sys.stdout.write('\r' + ' ' * 80 + '\r')

def main():
    parser = argparse.ArgumentParser(
        description="清理文本文件中的非 UTF-8 字符，支持大文件并显示进度。"
    )
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--chunk-size", type=int, default=1048576,
                        help="读取块大小（字节），默认1MB")
    parser.add_argument("--no-progress", action="store_true",
                        help="禁用进度显示")
    args = parser.parse_args()

    try:
        clean_utf8_file(args.input, args.output,
                        chunk_size=args.chunk_size,
                        show_progress=not args.no_progress)
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()