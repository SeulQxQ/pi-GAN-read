import logging
# from train import output_logs

def outlogs(message):

    # 配置日志系统，添加 FileHandler 来写入文件
    log_filename = 'test.txt'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_filename, 'a', 'utf-8')])

    # 记录一条信息级别的日志
    logging.info(message)

# output_log(f"output_01: {len(s)}, output_02: {len(s[0])}")
# output_log("output_02")
# output_log("output_03")
# logging.info(f"output_01: {len(s)}, output_02: {len(s[0])}")