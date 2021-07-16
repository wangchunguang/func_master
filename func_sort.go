package func_master

import (
	"fmt"
)

/*
对于golang的排序，其中数据量大于12的时候，如果当前递归深度大于2*ceil(lg(n+1))使用，堆排序，否则快排，分区点使用的三数取中法
小于等于12的时候，以6为间隔做一遍希尔排序，使大致有序，再做一遍插入排序
*/

// BubbleSort 冒泡排序
func BubbleSort(arr []int) []int {
	num := len(arr)
	if num <= 1 {
		return arr
	}
	for i := 0; i < num; i++ {
		//	 提前退出冒泡排序
		flag := false
		for j := 0; j < num-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j] // 交换
				flag = true
			}
		}
		if !flag { // 没有数据交换之后 直接退出
			break
		}
	}
	return arr
}

// InsertionSort 插入排序
func InsertionSort(arr []int) []int {
	num := len(arr)
	if num <= 1 {
		return arr
	}
	for i := 1; i < num; i++ {
		// value左边是已经排序好了的数，value是要进行排序的数
		value := arr[i]
		k := i - 1
		//	 寻找插入位置 同时顺序的移动数据位置
		for k >= 0 && arr[k] > value {
			// 左边的有序集合，如果插入的比左边的数大，那个就一向右边移动
			arr[k+1] = arr[k]
			k -= 1
		}
		// 将比较的数插入到指定的位置  因为for里面有一个k-=1  所以这里的是k+1
		arr[k+1] = value
	}
	return arr
}

// MergeSort 归并排序 start 表示开始排序的数组下标 end表示结束排序的数组下标,
func MergeSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}
	//	获取排序数据的中间值
	mid := len(arr) / 2
	//	 左边数组的大小 不包括mid
	left := MergeSort(arr[:mid])
	// 右边数组的大小 包括mid
	right := MergeSort(arr[mid:])
	return merge(left, right)
}

// 将并归排序的两个数组进行合并
func merge(left, right []int) []int {
	l, r := 0, 0
	arr := []int{}
	for l < len(left) && r < len(right) {
		if left[l] < right[r] {
			arr = append(arr, left[l])
			l++
		} else {
			arr = append(arr, right[r])
			r++
		}
	}
	arr = append(arr, left[l:]...)
	arr = append(arr, right[r:]...)
	return arr
}

// QuickSort 快速排序
func QuickSort(arr []int, left, right int) {
	if right > left {
		//	 位置划分
		pivot := partition(arr, left, right)
		// 左边排序
		QuickSort(arr, left, pivot-1)
		// 右边排序
		QuickSort(arr, pivot+1, right)
	}
}

func partition(arr []int, left, right int) int {
	// 导致left的位置为空
	pivot := arr[left]
	for left < right {
		//当右边的数小于挖坑的那个数，就停止，每次循环后边的数向左边的数递减
		for left < right && pivot <= arr[right] {
			right--
		}
		// 因为是挖坑的方法，所以左边left位置的数是空的，将右边和左边的数交换位置
		arr[left] = arr[right]
		//	left的指针值>pivot left指针像右移动，当遇见左边的数大于挖坑的数就停止
		for left < right && pivot >= arr[left] {
			left++
		}
		// 将左边的数和右边的数进行交换，又开始新一轮的循环比较
		arr[right] = arr[left]
	}
	// 循环结束，将挖坑的数放入该放入的位置
	arr[left] = pivot
	return left
}

// CountSort 计数排序 数据都大于0
func CountSort(arr []int) []int {
	num := len(arr)
	if num <= 1 {
		return nil
	}
	// 获取数据的范围
	max := arr[0]
	for i := 1; i < num; i++ {
		if max < arr[i] {
			max = arr[i]
		}
	}

	//	因为是计数排序，所以申请的一个数组数量为排序数组数量的大小
	countarr := make([]int, max+1)
	//	通过计数方式，先将计数中的每个桶进行初始化
	for i := 0; i <= max; i++ {
		//	每个桶中的数据为0
		countarr[i] = 0
	}
	//	 将每个元素的个数放入数组中
	for i := 0; i < num; i++ {
		// 比如 arr[i] = 5 那么countarr数组中第五个下标的数量就+1
		countarr[arr[i]]++
	}
	fmt.Println(countarr)
	//	 依次累加 累加的就是计算之后的数据，假如下标0的原始数据有1，下标1的数据有5个，那个下标1就是6
	for i := 1; i <= max; i++ {
		countarr[i] = countarr[i-1] + countarr[i]
	}
	//	临时数组 存储之后排序的结果
	stortarr := make([]int, num)
	// 计算排序的关键步骤  倒叙具有稳定性
	for i := num - 1; i >= 0; i-- {
		//原始数组从后面向前扫描，扫描到的数在countarr数组里面可以获取前面还有多少的数因为有下标0 所以-1，然后然后将数据添加到固定的位置
		// 例如 arr[i]下标的数是8 在countarr 发现8的前面还有16个数，所以stortarr【15】 = 8
		index := countarr[arr[i]] - 1
		stortarr[index] = arr[i]
		countarr[arr[i]]--
	}
	return stortarr
}
