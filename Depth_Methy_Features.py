# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import pysam
from scipy import stats
import os
import time


class DepthCalculator:
    """
    计算指定染色体或区域的覆盖度、GC含量，并提供类似 align_bs 的
    甲基化惩罚机制，可根据 CIGAR 运算符和碱基比对方式计算每条读段的惩罚分数。
    """

    def __init__(self,
                 filename,
                 reference_filename=None,
                 cpg_island_file=None,
                 bin_size=100,
                 region=None,
                 mismatch_penalty=4,
                 go_penalty=6,
                 ge_penalty=1,
                 unaligned_penalty=999999999,
                 methylation_processor=None):
        """
        参数说明
        ----------
        filename : str
            BAM/SAM 文件路径
        reference_filename : str
            参考基因组 (FASTA) 文件路径
        cpg_island_file : str, optional
            CpG island 文件（BED 等格式），可为空
        bin_size : int
            将染色体分块的块大小
        region : tuple or None
            指定处理的 (start, end) 区间，如为 None 则处理整条染色体
        mismatch_penalty : int
            普通碱基错配的惩罚分值
        go_penalty : int
            gap open 惩罚
        ge_penalty : int
            gap extension 惩罚
        unaligned_penalty : int
            无法比对或无效读段时的惩罚分值
        methylation_processor : MethylationProcessor, optional
            提供甲基化位点信息的对象，如果提供，将仅提取当前区域的甲基化信息
        """
        self.logger = logging.getLogger("methylcnv.depth")
        self.filename = filename
        self.reference_filename = reference_filename
        self.cpg_island_file = cpg_island_file
        self.bin_size = bin_size
        self.region = region
        if region is None:
            raise ValueError("region cannot be None. Please specify a region to process.")
        self.start, self.end = region

        self.mismatch_penalty = mismatch_penalty
        self.go_penalty = go_penalty
        self.ge_penalty = ge_penalty
        self.unaligned_penalty = unaligned_penalty
        
        # 存储当前区域的CpG位点信息
        self.methylation_processor = methylation_processor
        self.cpg_sites = {}
        self.cpg_meth_levels = {}
        
        # 仅当提供了methylation_processor时，为当前区域提取甲基化数据
        if self.methylation_processor:
            self._prepare_region_methylation_data()

        # 打开 BAM/SAM 文件
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".bam":
                self.file = pysam.AlignmentFile(filename, "rb",
                                                reference_filename=reference_filename if reference_filename else None)
            else:
                self.file = pysam.AlignmentFile(filename, "r",
                                                reference_filename=reference_filename if reference_filename else None)
        except Exception as e:
            self.logger.error(f"无法打开对齐文件 {filename}: {str(e)}")
            raise

        # 获取染色体信息
        self.references = self.file.references
        self.lengths = self.file.lengths
        self.chrom_lengths = dict(zip(self.references, self.lengths))

        # 打开参考基因组
        if reference_filename and os.path.isfile(reference_filename):
            try:
                self.fasta = pysam.FastaFile(reference_filename)
            except Exception as e:
                self.logger.error(f"无法打开参考基因组: {str(e)}")
                self.fasta = None
        else:
            self.fasta = None

        # 预备 CpG 位点集合
        self.cpg_ids = set()
        if self.fasta and self.cpg_island_file:
            self.load_reference_and_cpg()

    def _prepare_region_methylation_data(self):
        """
        从MethylationProcessor中仅提取当前区域的甲基化数据
        """
        if not self.methylation_processor or not self.region:
            return
            
        self.logger.info("开始从MethylationProcessor中提取当前区域的甲基化数据...")
        start_time = time.time()
        
        try:
            # 获取当前区域的区间
            region_start, region_end = self.region
            chrom = None  # 将在process_region中确定
            
            # 初始化染色体数据结构
            self.cpg_sites = {}
            self.cpg_meth_levels = {}
            
            # 我们会在process_region调用时设置具体的染色体
            self.logger.info(f"区域甲基化数据准备完毕，用时 {time.time() - start_time:.2f} 秒")
            
        except Exception as e:
            self.logger.error(f"处理区域甲基化数据时出错: {str(e)}")

    def extract_region_cpg_data(self, chr_name):
        """
        根据指定的染色体和当前区域提取CpG位点数据
        """
        if not self.methylation_processor:
            return
            
        start_time = time.time()
        region_start, region_end = self.region
        
        # 过滤当前区域的甲基化数据
        filtered_data = self.methylation_processor.meth_data[
            (self.methylation_processor.meth_data['Chr'] == chr_name) & 
            (self.methylation_processor.meth_data['Start'] >= region_start) & 
            (self.methylation_processor.meth_data['End'] <= region_end)
        ]
        
        # 初始化结构
        if chr_name not in self.cpg_sites:
            self.cpg_sites[chr_name] = []
            self.cpg_meth_levels[chr_name] = {}
        
        # 提取位点和甲基化水平
        for _, row in filtered_data.iterrows():
            pos = row['Start']
            meth_percent = row['MethPercent'] / 100.0  # 转换为0-1范围
            
            # 添加CpG位点位置和甲基化水平
            self.cpg_sites[chr_name].append(pos)
            self.cpg_meth_levels[chr_name][pos] = meth_percent
        
        # 排序以便后续查找
        if chr_name in self.cpg_sites:
            self.cpg_sites[chr_name].sort()
            
        self.logger.info(f"提取区域 {chr_name}:{region_start}-{region_end} 的甲基化数据，共 {len(self.cpg_sites.get(chr_name, []))} 个CpG位点，用时 {time.time() - start_time:.2f} 秒")

    def is_cpg_site(self, chrom, position):
        """
        检查给定位置是否为CpG位点
        """
        if self.methylation_processor:
            # 使用MethylationProcessor数据
            if chrom not in self.cpg_meth_levels:
                return False
            return position in self.cpg_meth_levels[chrom]
        else:
            # 使用传统方法
            return (chrom, position) in self.cpg_ids

    def get_methylation_level(self, chrom, position):
        """
        获取指定CpG位点的甲基化水平
        """
        if self.methylation_processor and chrom in self.cpg_meth_levels and position in self.cpg_meth_levels[chrom]:
            return self.cpg_meth_levels[chrom][position]
        
        # 找不到甲基化信息，返回默认值0.5
        return 0.5

    def load_reference_and_cpg(self):
        """
        根据 cpg_island_file 与参考基因组获取 CpG 位置信息，
        并存储在 self.cpg_ids 中 (chr_name, pos) 的形式
        """
        self.logger.info("加载 CpG 岛数据并扫描参考序列上的 CG...")
        cpg_island_regions = []
        try:
            with open(self.cpg_island_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    fields = line.strip().split()
                    if len(fields) < 3:
                        continue
                    c_chr = fields[0].replace("chr", "")
                    c_start = int(fields[1])
                    c_end = int(fields[2])
                    cpg_island_regions.append((c_chr, c_start, c_end))
        except Exception as e:
            self.logger.warning(f"读取 cpg_island_file 失败: {str(e)}")

        for chrom_name in self.references:
            try:
                seq_length = self.chrom_lengths[chrom_name]
                refseq = self.fasta.fetch(chrom_name, 0, seq_length).upper()
            except Exception as e:
                self.logger.warning(f"无法获取参考序列 {chrom_name}: {str(e)}")
                continue

            for i in range(len(refseq) - 1):
                if refseq[i] == 'C' and refseq[i + 1] == 'G':
                    self.cpg_ids.add((chrom_name, i))
                    self.cpg_ids.add((chrom_name, i + 1))

    def get_chromosome_length(self, chr_name):
        """获取指定染色体长度"""
        idx = self.references.index(chr_name)
        return self.lengths[idx]

    def read_chromosome(self, chr_name):
        """
        读取指定染色体或 region 范围内的覆盖度信息，并统计到 bin 中。
        返回 rd_p, rd_u, his_read_frg:
            rd_p: 每个 bin 的总覆盖度
            rd_u: 每个 bin 的高质量映射覆盖度 (mapping_quality>0)
            his_read_frg: 读长 vs 插入片段长度的直方图
        """
        if chr_name not in self.chrom_lengths:
            self.logger.warning(f"对齐文件中未找到染色体 '{chr_name}'。")
            return None, None, None

        if self.region:
            start, end = self.region
            length = end - start
            offset = start
        else:
            length = self.chrom_lengths[chr_name]
            offset = 0

        n = (length + self.bin_size - 1) // self.bin_size
        rd_p = np.zeros(n)
        rd_u = np.zeros(n)

        if self.region:
            fetch_start = max(0, start - 300)
            fetch_end = min(self.chrom_lengths[chr_name], end + 300)
            reads_iter = self.file.fetch(chr_name, fetch_start, fetch_end)
        else:
            reads_iter = self.file.fetch(chr_name)

        for r in reads_iter:
            if r.is_unmapped or r.is_secondary or r.is_duplicate:
                continue

            if self.region:
                overlap_start = max(start, r.reference_start)
                overlap_end = min(end, r.reference_end)
            else:
                overlap_start = r.reference_start
                overlap_end = r.reference_end

            # 检查reference_end是否为None
            if overlap_end is None:
                continue
                
            if overlap_start < overlap_end:
                start_bin = (overlap_start - offset) // self.bin_size
                end_bin = (overlap_end - offset - 1) // self.bin_size + 1
                start_bin = max(0, min(start_bin, n - 1))
                end_bin = max(0, min(end_bin, n))
                for bin_idx in range(start_bin, end_bin):
                    rd_p[bin_idx] += 1
                    if r.mapping_quality > 0:
                        rd_u[bin_idx] += 1

        return rd_p, rd_u, None

    def calculate_gc_content(self, chr_name, start, end):
        """
        计算指定区间的 GC 含量并按 bin_size 切分
        返回一个长度为 bin 数量的数组，每个元素为对应区间的 GC 百分比
        """
        if not self.fasta:
            return None
        try:
            sequence = self.fasta.fetch(chr_name, start, end).upper()
            n_bins = ((end - start) + self.bin_size - 1) // self.bin_size
            gc = np.zeros(n_bins)

            for i in range(n_bins):
                bin_start = i * self.bin_size
                bin_end = min(bin_start + self.bin_size, len(sequence))
                if bin_end <= bin_start:
                    break
                bin_seq = sequence[bin_start:bin_end]
                gc_count = bin_seq.count('G') + bin_seq.count('C')
                total_count = gc_count + bin_seq.count('A') + bin_seq.count('T')
                if total_count > 0:
                    gc[i] = (gc_count * 100.0) / total_count
            return gc
        except Exception as e:
            self.logger.error(f"在 {chr_name} 计算 GC 含量时出现错误: {str(e)}")
            return None

    def calc_base_nuc_penalty(self, chr_name, ref_pos, read_base):
        """
        根据align_bs.cpp的思路计算碱基惩罚分数：
        1. 如果碱基匹配，惩罚为0
        2. 如果在CpG位点:
           a) 碱基是甲基化的(未发生C→T或G→A转换)，惩罚为(1-methyl_ratio)*mismatch_penalty
           b) 碱基是未甲基化的(发生了C→T或G→A转换)，惩罚为methyl_ratio*mismatch_penalty
        3. 如果是常规的错配(非CpG位点或不涉及C→T或G→A转换)，惩罚为mismatch_penalty
        """
        if not self.fasta:
            return self.unaligned_penalty
            
        try:
            # 获取参考基因组上的碱基
            ref_base_str = self.fasta.fetch(chr_name, ref_pos, ref_pos + 1).upper()
        except:
            return self.unaligned_penalty

        if len(ref_base_str) == 0:
            return self.unaligned_penalty

        ref_b = ref_base_str[0]
        read_b = read_base.upper()
        
        # 步骤1: 检查碱基是否完全匹配
        if ref_b == read_b:
            return 0
            
        # 步骤2: 检查是否为CpG位点上的特殊转换
        is_cpg_site = self.is_cpg_site(chr_name, ref_pos)
        
        if is_cpg_site:
            # 甲基化状态判断逻辑
            is_ct_conversion = (ref_b == 'C' and read_b == 'T')
            is_ga_conversion = (ref_b == 'G' and read_b == 'A')
            
            # 判断是甲基化还是未甲基化
            is_methylated = not (is_ct_conversion or is_ga_conversion)
            
            if (ref_b == 'C' or ref_b == 'G'):  # 只关注C和G位点
                methyl_ratio = self.get_methylation_level(chr_name, ref_pos)
                
                if is_methylated:
                    # 甲基化的碱基，使用(1-methyl_ratio)计算惩罚
                    return (1 - methyl_ratio) * self.mismatch_penalty
                else:
                    # 未甲基化的碱基，使用methyl_ratio计算惩罚
                    return methyl_ratio * self.mismatch_penalty
        
        # 步骤3: 如果既不是匹配也不是CpG位点上的特殊转换，则是常规错配
        return self.mismatch_penalty

    def calculate_methylation_penalty(self, chr_name):
        """
        基于对齐结果与参考序列，逐条读段计算惩罚分数，并按 bin 聚合。
        """
        if self.region:
            start, end = self.region
            length = end - start
            offset = start
            n_bins = (length + self.bin_size - 1) // self.bin_size
            penalty_by_bin = np.zeros(n_bins)
            fetch_start = max(0, start)
            fetch_end = min(self.chrom_lengths[chr_name], end)
            read_iter = self.file.fetch(chr_name, fetch_start, fetch_end)
        else:
            length = self.chrom_lengths[chr_name]
            offset = 0
            n_bins = (length + self.bin_size - 1) // self.bin_size
            penalty_by_bin = np.zeros(n_bins)
            read_iter = self.file.fetch(chr_name)

        self.logger.info("开始计算甲基化惩罚分数...")
        start_time = time.time()

        def calc_read_penalty(aln):
            if aln.is_unmapped:
                return self.unaligned_penalty
            penalty_sum = 0
            ref_idx = aln.reference_start
            seq = aln.query_sequence
            read_idx = 0
            ctups = aln.cigartuples
            if not ctups:
                return self.unaligned_penalty

            for op, length_op in ctups:
                if op == 0:  # M (match/mismatch)
                    for _ in range(length_op):
                        if read_idx >= len(seq):
                            penalty_sum += self.unaligned_penalty
                            return penalty_sum
                        penalty_sum += self.calc_base_nuc_penalty(chr_name, ref_idx, seq[read_idx])
                        read_idx += 1
                        ref_idx += 1
                elif op == 1:  # insertion
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    read_idx += length_op
                elif op == 2:  # deletion
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    ref_idx += length_op
                else:
                    if op in [4, 5]:
                        read_idx += length_op
                    elif op == 3:
                        ref_idx += length_op
            return penalty_sum

        count = 0
        for aln in read_iter:
            if count % 10000 == 0:
                self.logger.debug(f"已处理 {count} 条读段")
            count += 1
            
            if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
                continue
                
            penalty_val = calc_read_penalty(aln)
            overlap_start = aln.reference_start
            overlap_end = aln.reference_end
            
            # 检查reference_end是否为None
            if overlap_start is None or overlap_end is None:
                continue
            
            if self.region:
                start, end = self.region
                overlap_start = max(start, overlap_start)
                overlap_end = min(end, overlap_end)

            if overlap_start < overlap_end:
                start_bin = (overlap_start - offset) // self.bin_size
                end_bin = (overlap_end - offset - 1) // self.bin_size + 1
                start_bin = max(0, min(start_bin, n_bins - 1))
                end_bin = max(0, min(end_bin, n_bins))
                for b in range(start_bin, end_bin):
                    penalty_by_bin[b] += penalty_val
        
        self.logger.info(f"甲基化惩罚计算完成，共处理 {count} 条读段，用时 {time.time() - start_time:.2f} 秒")
        return penalty_by_bin

    def process_region(self, chr_name):
        """
        核心入口函数：
        1) 读取染色体/区间内的覆盖度
        2) 计算 GC 含量
        3) 计算类似 align_bs 的甲基化惩罚 penalty_by_bin
        返回一个包含这些结果的字典
        """
        # 如果提供了MethylationProcessor，先为当前区域提取甲基化数据
        if self.methylation_processor:
            self.extract_region_cpg_data(chr_name)
            
        self.logger.info(f"开始处理区域 {chr_name}:{self.region[0]}-{self.region[1]}")
        start_time = time.time()
            
        rd_p, rd_u, _ = self.read_chromosome(chr_name)
        if rd_p is None:
            return None

        results = {
            'rd_p': rd_p,
            'rd_u': rd_u,
        }

        if self.region:
            start, end = self.region
            gc_vals = self.calculate_gc_content(chr_name, start, end)
        else:
            length = self.chrom_lengths[chr_name]
            gc_vals = self.calculate_gc_content(chr_name, 0, length)
        results['gc_content'] = gc_vals

        penalty_by_bin = self.calculate_methylation_penalty(chr_name)
        if penalty_by_bin is not None:
            results['methyl_penalty'] = penalty_by_bin

            # === 在此增加归一化举例 ===
            # 例如除以 (rd_p + 1)
            norm_penalty = penalty_by_bin / (rd_p + 1)
            results['methyl_penalty_norm'] = norm_penalty
            
            # === 添加深度-甲基化比例特征 ===
            if 'meth_density' in results and 'meth_variance' in results:
                # 添加深度与甲基化方差的比例特征
                epsilon = 1e-6  # 避免除零错误
                results['depth_var_ratio'] = rd_p / (results['meth_variance'] + epsilon)
                
                # 添加深度与甲基化密度的比例特征
                results['depth_density_ratio'] = rd_p / (results['meth_density'] + epsilon)
                
                # 添加甲基化惩罚与甲基化方差的比例
                results['penalty_var_ratio'] = penalty_by_bin / (results['meth_variance'] + epsilon)
            
            # === 添加局部上下文特征 ===
            # 计算深度变化率（滑动差分）
            depth_gradient = np.zeros_like(rd_p)
            depth_gradient[1:] = np.diff(rd_p)  # 相邻bin深度差异

            # 计算滑动窗口深度均值和标准差
            window_size = 5  # 5个bin的窗口
            depth_smooth = np.zeros_like(rd_p)
            depth_local_std = np.zeros_like(rd_p)

            for i in range(len(rd_p)):
                window_start = max(0, i - window_size // 2)
                window_end = min(len(rd_p), i + window_size // 2 + 1)
                window_data = rd_p[window_start:window_end]
                depth_smooth[i] = np.mean(window_data)
                depth_local_std[i] = np.std(window_data)

            # 添加到结果
            results['depth_gradient'] = depth_gradient
            results['depth_smooth'] = depth_smooth
            results['depth_local_std'] = depth_local_std

            # 检测深度突变点（CNV边界候选）
            depth_change_score = np.abs(depth_gradient) / (depth_smooth + 1e-6)
            results['depth_change_score'] = depth_change_score
            
            # === 获取并添加甲基化处理器的特征 ===
            if self.methylation_processor:
                meth_features = self.methylation_processor.process_chromosome(chr_name)
                if meth_features:
                    for key, value in meth_features.items():
                        # 把甲基化特征直接合并到results中
                        if key not in ['bin_coords', 'bin_start', 'chrom']:
                            results[key] = value
            
            # === 添加甲基化相关的局部上下文特征 ===
            if 'meth_variance' in results:
                # 计算甲基化方差的梯度
                meth_var = results['meth_variance']
                meth_var_gradient = np.zeros_like(meth_var)
                meth_var_gradient[1:] = np.diff(meth_var)
                
                # 滑动窗口处理
                meth_var_smooth = np.zeros_like(meth_var)
                for i in range(len(meth_var)):
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(meth_var), i + window_size // 2 + 1)
                    window_data = meth_var[window_start:window_end]
                    meth_var_smooth[i] = np.mean(window_data)
                
                # 甲基化与深度变化的相关性特征
                if len(meth_var_gradient) != len(depth_gradient):
                    self.logger.warning(f"维度不匹配：depth_gradient({len(depth_gradient)})与meth_var_gradient({len(meth_var_gradient)})")
                    # 保留较小维度，裁剪较大维度
                    if len(meth_var_gradient) > len(depth_gradient):
                        meth_var_gradient = meth_var_gradient[:len(depth_gradient)]
                    else:
                        depth_gradient = depth_gradient[:len(meth_var_gradient)]
                
                coherence_score = depth_gradient * meth_var_gradient
                
                # 添加到结果
                results['meth_var_gradient'] = meth_var_gradient
                results['meth_var_smooth'] = meth_var_smooth
                results['depth_meth_coherence'] = coherence_score
            
        # 增加附加信息
        if self.region:
            bin_count = len(rd_p)
            results['bin_start'] = [self.region[0] + i * self.bin_size for i in range(bin_count)]
            results['chrom'] = [chr_name] * bin_count
            
        self.logger.info(f"区域处理完成，用时 {time.time() - start_time:.2f} 秒")
        return results

    def close(self):
        """关闭相关文件句柄"""
        try:
            if self.file:
                self.file.close()
        except:
            pass
        try:
            if self.fasta:
                self.fasta.close()
        except:
            pass


class MethylationProcessor:
    """
    读取外部甲基化信息文件（例如 MethylDackel 输出），并可生成 bin 级别的特征。
    """

    def __init__(self,
                 meth_file,
                 reference_filename=None,
                 bin_size=100,
                 region=None):
        self.meth_file = meth_file
        self.reference_file = reference_filename
        self.bin_size = bin_size
        self.region = region
        self.logger = logging.getLogger("methylcnv.methylation")

        if reference_filename and os.path.isfile(reference_filename):
            self.fasta = pysam.FastaFile(reference_filename)
        else:
            self.fasta = None

        self.meth_data = pd.read_csv(
            meth_file, sep='\t',
            dtype={
                'Chr': str,
                'Start': int,
                'End': int,
                'MethPercent': float,
                'MethCount': int,
                'UnmethCount': int
            }
        )
        self.meth_data['Chr'] = self.meth_data['Chr'].str.replace('chr', '')

    def process_chromosome(self, chrom):
        if self.fasta:
            chrom_length = self.fasta.get_reference_length(chrom)
        else:
            chrom_length = None

        chr_data = self.meth_data[self.meth_data['Chr'] == chrom].copy()
        if len(chr_data) == 0:
            self.logger.warning(f"未在 {chrom} 找到甲基化数据。")
            return None

        if self.region and chrom_length is not None:
            start, end = self.region
            chr_data = chr_data[(chr_data['Start'] >= start) & (chr_data['End'] <= end)]
            region_length = end - start
        elif chrom_length is not None:
            start = 0
            end = chrom_length
            region_length = end - start
        else:
            start = chr_data['Start'].min()
            end = chr_data['End'].max()
            region_length = end - start

        num_bins = (region_length + self.bin_size - 1) // self.bin_size
        features = {
            'meth_level': np.zeros(num_bins),
            'meth_density': np.zeros(num_bins),
            'meth_variance': np.zeros(num_bins),
            'coverage': np.zeros(num_bins),
            'meth_entropy': np.zeros(num_bins),
            'bin_coords': []
        }

        chr_data['rel_start'] = chr_data['Start'] - start
        chr_data['bin_idx'] = chr_data['rel_start'] // self.bin_size

        for bin_idx in range(num_bins):
            bin_start = start + bin_idx * self.bin_size
            bin_end = min(bin_start + self.bin_size, end)
            features['bin_coords'].append((bin_start, bin_end))

            bin_data = chr_data[chr_data['bin_idx'] == bin_idx]
            if len(bin_data) > 0:
                coverage_arr = bin_data['MethCount'] + bin_data['UnmethCount']
                if coverage_arr.sum() > 0:
                    features['meth_level'][bin_idx] = np.average(
                        bin_data['MethPercent'],
                        weights=coverage_arr
                    )
                features['meth_density'][bin_idx] = len(bin_data) / self.bin_size
                features['meth_variance'][bin_idx] = np.var(bin_data['MethPercent'])
                features['coverage'][bin_idx] = np.mean(coverage_arr)

                total_meth = bin_data['MethCount'].sum()
                total_unmeth = bin_data['UnmethCount'].sum()
                total_sum = total_meth + total_unmeth
                if total_sum > 0:
                    from scipy.stats import entropy
                    dist_norm = np.array([total_meth, total_unmeth]) / total_sum
                    features['meth_entropy'][bin_idx] = entropy(dist_norm)
                    
        # 添加bin_start和chrom信息
        bin_starts = [start + i * self.bin_size for i in range(num_bins)]
        features['bin_start'] = bin_starts
        features['chrom'] = [chrom] * num_bins

        # 删除不再需要的 'bin_coords'
        del features['bin_coords']

        return features 