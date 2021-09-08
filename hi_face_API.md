<h1 style="text-align:center"> 人脸搜索服务API文档 <h1>

- - - 


<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=3 orderedList=false} -->

<!-- code_chunk_output -->

- [说明](#说明)
- [1-人脸库操作](#1-人脸库操作)
  - [1.1-获取人脸库信息](#11-获取人脸库信息)
  - [1.2-创建人脸库](#12-创建人脸库)
  - [1.3-删除人脸库](#13-删除人脸库)
  - [1.4-修改人脸库容量大小](#14-修改人脸库容量大小)
- [2-人脸操作](#2-人脸操作)
  - [2.1-添加人脸](#21-添加人脸)
  - [2.2-删除人脸](#22-删除人脸)
  - [2.3-利用图片进行人脸搜索](#23-利用图片进行人脸搜索)
  - [2.4-查询指定人脸库内某条特征信息](#24-查询指定人脸库内某条特征信息)
  - [2.5-使用特征字符串插入特征信息](#25-使用特征字符串插入特征信息)
  - [2.6-使用特征字符串进行人脸搜索](#26-使用特征字符串进行人脸搜索)
  - [2.7-给定一张图片获取人脸特征字符串](#27-给定一张图片获取人脸特征字符串)
  - [2.8-给定图片和库内feature_id进行人脸对比验证](#28-给定图片和库内feature_id进行人脸对比验证)
  - [2.9-按时间段删除向量](#29-按时间段删除向量)
- [3. 其他](#3-其他)
  - [3.1-人脸检测](#31-人脸检测)
  - [3.2-人脸活体验证登录](#32-人脸活体验证登录)
  - [3.3-人脸属性估计](#33-人脸属性估计)
  - [3.4-服务状态检测](#34-服务状态检测)
  - [3.5-口罩检测](#35-口罩检测)
  - [3.6-人证对比](#36-人证对比)

<!-- /code_chunk_output -->


- - - 

<div STYLE="page-break-after: always;"></div>

## 说明
  当前可访问该地址进行请求操作http://183.252.15.157:7006

## 1-人脸库操作
### 1.1-获取人脸库信息
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/info
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|否|人脸库名称，不填写或为空字符时返回所有人脸库信息|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空|
|result|json 数组|返回结果，json对象数组|

**result中字段说明**
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|size|Integer|人脸库现存人脸数量|
|max_size|Integer|人脸库可存储的最大人脸数量|
|info|String|人脸库备注信息|
|is_multiple_gpus|Bool|人脸库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|

#### 请求示例
~~~http
POST /v1/db/info HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": ""
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "info": "test0",
      "db_name": "database_0",
      "max_size": 10000,
      "size": 6,
      "is_multiple_gpus": false
    },
    {
      "info": "test1",
      "db_name": "database_1",
      "max_size": 10000,
      "size": 6,
      "is_multiple_gpus": true
    }
  ]
}
~~~

### 1.2-创建人脸库
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/create
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称，唯一标识|
|max_size|Integer|是|人脸库容量，范围限制1~30000000|
|is_multiple_gpus|Bool|否|人脸库是否使用多GPU,默认False,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|否|人脸库备注信息，默认空字符|


#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空字符|
|result|json数组|返回结果|

result中字段说明：
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|size|Integer|人脸库现存人脸数量|
|max_size|Integer|人脸库可存储的最大人脸数量|
|is_multiple_gpus|Bool|人脸库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|人脸库备注信息|


#### 请求示例
~~~http
POST /v1/db/create HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_2",
  "max_size": 10000,
  "is_multiple_gpus": false,
  "info": "database test 2"
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 10000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

### 1.3-删除人脸库
#### 请求方式
- POST
#### 请求URL
    http；//{host}:{post}/v1/db/remove
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|size|Integer|人脸库的大小|
|max_size|Integer|人脸库最大容量|
|is_multiple_gpus|Bool|人脸库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|人脸库的备注信息|
#### 请求示例
~~~http
POST /v1/db/remove HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_2"
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 10000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

### 1.4-修改人脸库容量大小
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/update
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|max_size|Integer|是|人脸库容量，范围限制1~1000000|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|size|Integer|人脸库的大小|
|max_size|Integer|人脸库最大容量|
|info|String|人脸库的备注信息|
|is_multiple_gpus|Bool|人脸库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|

#### 请求示例
~~~http
POST /v1/db/update HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_0",
  "max_size": 1000
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 1000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

## 2-人脸操作
### 2.1-添加人脸
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/insert
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature_id|Integer|是|人脸唯一特征ID|
|image|String|是|人脸图片的base64编码字符串|
|info|String|否|人脸备注信息，默认空字符|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空，未检测到人脸也会返回错误|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|备注信息|
#### 请求示例
~~~http
POST /v1/db/insert HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_1",
  "feature_id": 100,
  "info": "test face",
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe..."
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591,...]",
      "feature_id": 100,
      "info": "test face"
    }
  ]
}
~~~

### 2.2-删除人脸
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/delete
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature_id|integer|是|人脸唯一特征ID|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|备注信息|
#### 请求示例
~~~http
POST /v1/db/delete HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
	"db_name": "database_2",
	"feature_id": 22
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
      "feature_id": 22,
      "info": "test face"
    }
  ]
}
~~~

### 2.3-利用图片进行人脸搜索
#### 请求方式
- POST
#### 请求URL
    http:{host}:{port}/v1/db/search
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|图片base64编码字符串|
|db_name|String|是|人脸库名称|
|top|integer|否|返回相似度最高的前top个结果，默认5|
|nprobe|Integer|否|聚簇搜索数量，1~128，默认128，越大搜索越精确，相对速度会更慢|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|备注信息|
|score|Float|相似度 0~1，1为完全相似|

#### 请求示例
~~~http
POST /v1/db/search HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
	"db_name": "database_1",
	"top": 2,
	"image": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 100,
      "info": "test face",
      "socre": 0.9676
    },
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 200,
      "info": "test face 2",
      "socre": 0.9376
    }
  ]
}
~~~

### 2.4-查询指定人脸库内某条特征信息
#### 请求方式
- POST
#### 请求URL
    http:{host}:{port}/v1/db/get
#### Headers
    Content-Type	application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature_id|Integer|是|特征ID|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|info|String|备注信息|

#### 请求示例
~~~http
POST /v1/db/get HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_1",
  "feature_id": 100
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
      "feature_id": 100,
      "info": "test face"
    }
  ]
}
~~~

### 2.5-使用特征字符串插入特征信息
#### 请求方式
- POST
#### 请求URL
    http:{host}:{port}/v1/db/insert_feature
#### Headers
    Content-Type	application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature_id|Integer|是|特征ID|
|faceCoord|Array|是|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|feature|String|是|特征字符串|
|info|String|否|备注信息|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|info|String|备注信息|

#### 请求示例
~~~http
POST /v1/db/insert_feature HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_1",
  "faceCoord" : [100, 120, 200, 220],
  "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
  "feature_id": 100,
  "info": "test face"
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
  {
    "db_name": "database_1",
    "faceCoord" : [100, 120, 200, 220],
    "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
    "feature_id": 100,
    "info": "test face"
  }
  ]
}
~~~

### 2.6-使用特征字符串进行人脸搜索
#### 请求方式
- POST
#### 请求URL
    http:{host}:{port}/v1/db/search_feature
#### Headers
    Content-Type	application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature|String|是|特征字符串|
|top|integer|否|返回相似度最高的前top个结果，默认5|
|nprobe|Integer|否|聚簇搜索数量，1~128，默认128，越大搜索越精确，相对速度会更慢|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|score|Float|相似度 0~1，1为完全相似|
|faceCoord|Array|人脸框位置 [x_min. y_min, x_max, y_max]|
|info|String|备注信息|

#### 请求示例
~~~http
POST /v1/db/search_feature HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_1",
  "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
  "top": 2
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 100,
      "info": "test face",
      "socre": 0.9676
    },
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 200,
      "info": "test face 2",
      "socre": 0.9376
    }
  ]
}
~~~

### 2.7-给定一张图片获取人脸特征字符串
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/extract_feature

#### Headers
    Content-Type	application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|图片base64编码字符串|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|正常则返回空, 图片检测不到人脸会报错|
|result|json数组|返回结果|

result中的字段说明
|字段名|类型|描述|
|---|---|---|
|face_coord|Array|人脸框位置，[x_min, y_min, x_max, y_max]|
|feature|String|人脸特征字符串|


#### 请求示例
~~~http
POST /face/get_feature HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe..."
}
~~~
#### 返回结果示例
~~~json
{
    "error": "",
    "result": [
        {
            "face_coord": [396, 623, 413, 656],
            "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]"
        }
    ]
}
~~~

### 2.8-给定图片和库内feature_id进行人脸对比验证
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/v1/db/face/verify
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|feature_id|Integer|是|需要进行对比的库中人脸ID|
|image|String|是|人脸图片的base64编码字符串|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|人脸库名称|
|feature_id|Integer|特征ID|
|face_coord_0|Array|库内人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|face_coord_1|Array|给定图片的人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|库内人脸备注信息|
|is_match|Bool|两张人脸是否是一个人，同一人则True|
|similar_score|Float|相似度，0~1|
#### 请求示例
~~~http
POST /v1/db/face/verify HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "db_name": "database_1",
  "feature_id": 100,
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe..."
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "db_name": "database_1",
      "feature_id": 100,
      "info": "",
      "face_coord_0": [98, 22, 112, 168],
      "face_coord_1": [10, 20, 110, 120],
      "is_match": true,
      "similar_score": 0.95
    }
  ]
}
~~~

### 2.9-按时间段删除向量
#### 请求方式
- POST or DELETE
#### 请求URL
    http://{host}:{port}/v1/db/delete_by_date
#### Headers
    Content-Type    application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|人脸库名称|
|begin_time|String|是|开始日期时间，格式为'%Y-%m-%d %H:%M:%S',如 '2021-03-13 10:30:30'|
|end_time|String|是|结束日期时间，格式为'%Y-%m-%d %H:%M:%S', 如 '2021-05-19 10:30:30'|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空|
|result|json数组|返回的结果|

result中的字段说明
|字段名|类型|描述|
|---|---|---|
|total|Integer|删除向量的总数量|
|feature_ids|Array|删除的人脸id|

#### 请求示例
~~~http
POST /faiss/feature/delete_by_date HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
    "db_name": "cbvsp",
    "begin_time": "2020-10-20 12:10:10",
    "end_time": "2021-01-02 12:10:10"
}
~~~

#### 返回结果示例
~~~json
{
    "error": "",
    "result": [
        {
            "total": 3,
            "feature_ids": [1, 2, 3]
        }
    ]
}
~~~


## 3. 其他
### 3.1-人脸检测
上传一张图片，返回图片中人脸的位置坐标
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/detect
#### Headers
    Content-Type  application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|图片的base64编码字符串|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|face_coord|数组|人脸位置坐标，（左上角x, 左上角y,右下角x, 右下角y）,空数组表示没检测到人脸|

#### 请求示例
~~~http
POST /face/detect   HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe..."
}
~~~

#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "face_coord": [[10, 10, 110, 110], [20, 20, 120, 120]]
    }
  ]
}
~~~

### 3.2-人脸活体验证登录
上传一张人脸图片，首先会判断该人脸是真人还是假人（图片欺骗），若为真人则与指定人脸库中人脸进行对比，返回对比匹配的结果
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/anti/detect

#### Headers
    Content-Type	application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|图片base64编码字符串。当设备为"computer"时，图片像素尺寸应为H=720 W=1280，且人脸位于中间700x700区域；当设备为"phone"时，图片像素尺寸应为H=960 W=720，且人脸位于中间700区域内，若人脸超过限定区域则会被判定为人脸不存在|
|db_name|String|否|人脸库名称，默认 ‘face_login_test'|
|device|String|否|"computer" or "phone",默认 "computer"。当设备为电脑时接收图片的像素为H=720 W=1280, 当设备为手机时接收图片的像素为H=960 W=720，|
|feature_id|Integer|否|人脸对应的特征id，默认-1，即不指定。当指定了特征id时，进行1：1对比，不指定时进行全库1：N对比|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|正常则返回空, 否则返回相应错误信息|
|result|json数组|返回结果|

result中的字段说明
|字段名|类型|描述|
|---|---|---|
|real_face|Bool|是否为真人，真人返回True,假人返回False|
|score|Float|该图片为真人/假人的置信度|
|box|Array|人脸框 [right_top_x,  right_top_y, left_bottom_x, left_bottom_y]|
|db_name|String|进行搜索的人脸库名称|
|feature_id|Integer|特征ID，假人或全库搜索没查找到匹配的人时返回-1|
|similar_score|Float|人脸相似度，假人或全库搜索没查找到匹配的人时返回-1，当指定特征id进行1：1对比时，会返回0~1之间的浮点数，当该值大于0.87时，认为两张图片是同一个人|


#### 请求示例
~~~http
POST /face/anti/detect HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe...",
  "db_name": "face_login_test"
}
~~~
#### 返回结果示例
~~~json
{
    "error": "",
    "result": [
        {
            "box": [396, 623, 413, 656],
            "real_face": true,
            "score": 0.811273747457,
            "db_name": "face_login_test",
            "feature_id": 2764236875682,
            "similar_score": 0.90
        }
    ]
}
~~~
#### 注意
为了进一步提高人脸活体检测的安全性，建议拍摄采样多张人脸图片，然后连续调用该接口，当连续二到三次判断为真人时，予以通过

### 3.3-人脸属性估计
上传一张人脸图片，返回相关人脸信息，包括人脸位置坐标、年龄、性别、角度、人种、表情、眼镜、帽子、口罩、质量
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/attribute/detect

#### Headers 
    Content-Type  application/json

#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|人脸图片base64编码字符串|
|detection_threshold|Float|否|人脸检测阈值，0~1，默认0.95|
|min_size|Integer|否|人脸最小检测尺寸，30~500，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|正常则返回空, 图片检测不到人脸会报错|
|result|Array|返回结果,人脸属性信息|

result中的字段说明
|字段名|类型|描述|
|---|---|---|
|face_coord|Array|人脸位置坐标，[right_top_x, right_top_y, left_bottom_x, left_bottom_y]|
|pitch|Float|俯仰角，头部上下转动角度|
|yaw|Float|偏航角，头部左右转动角度|
|roll|Float|滚动角，歪头角度|
|mask|String|'none'：无口罩 or 'mask'：有口罩 |
|mask_probability|Float|口罩置信度，已进行归一映射， 大于0.90 认为可信|
|glasses|String|'none': 无眼镜, 'normal':普通眼镜, 'dark':墨镜 |
|glasses_probability|Float|眼镜置信度，未进行归一映射，暂时无参考意义|
|quality|Float|人脸质量，0-1，小于0.3认为人脸质量很差|
|hat|String|'none' or 'hat' |
|hat_probability|Float|帽子置信度，已进行归一映射，大于0.90认为可信|
|age|Integer|年龄|
|gender|String|性别 'female' or 'male'|
|gender_probability|Float|性别置信度，已进行归一化映射，大于0.90认为可信|
|race|String|'white':白人，'asian':黄种人（东亚）， 'indian': 印度人，'black':黑人 |
|race_probability|Float|种族置信度，未进行归一化映射，暂时无参考意义|
|expression|String|'neutral':中性， 'happy':开心， 'sad':伤心， 'surprise':惊讶，'fear':害怕， 'disgust':厌恶， 'anger': 生气 |
|expression_probability|Float|表情置信度，未进行归一化映射，暂时无参考意义|

#### 请求示例
~~~http
POST /face/attribute/detect HTTP/1.1
Host: 192.168.96.136:7006
Content-Type: application/json

{
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe...",
}
~~~
#### 返回结果示例
~~~json
{
    "error": "",
    "result": [
        {
          "face_coord": [396, 623, 413, 656],
          "pitch": -7.047977447509766,
          "yaw": -1.047977447509766,
          "roll": 3.047977447509766,
          "mask": "none",
          "mask_probability": 0.93,
          "glasses": "normal",
          "glasses_probability": 0.93,
          "quality": 0.60,
          "hat": "none",
          "hat_probability": 0.93,
          "age": 24,
          "gender": "female",
          "gender_probability": 0.93,
          "race": "asian",
          "race_probability": 0.93,
          "expression": "neutral",
          "expression_probability": 0.93
        }
    ]
}
~~~

### 3.4-服务状态检测
#### 请求方式
- GET

#### 请求URL
    http://{host}:{port}/face_rec/state

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|state|Integer| 0或1，1表示服务正常，0表示服务出现异常|

### 3.5-口罩检测
上传一张图片，检测图片中人脸是否戴口罩
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/mask
#### Headers
    Content-Type  application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image|String|是|图片的base64编码字符串，宽：480，高：640|
|threshold|float|否|人脸检测的阈值，0~1，默认0.95|
|size|integer|否|最小的人脸尺寸，30~300，默认50|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，没检测到人脸也会返回错误，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|face_coord|数组|人脸位置坐标，（左上角x, 左上角y,右下角x, 右下角y）|
|mask|bool| true 为戴口罩，false为未戴口罩|

#### 请求示例
~~~http
POST /face/mask   HTTP/1.1
Host: 183.252.15.157:7006
Content-Type: application/json

{
  "image": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe...",
  "threshold": 0.97,
  "size": 60
}
~~~

#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
        "face_coord": [10, 10, 110, 110],
        "mask": true
    },
    {
        "face_coord": [100, 100, 200, 200],
        "mask": false
    }
  ]
}
~~~

### 3.6-人证对比
上传证件照片和抓拍人脸图片进行对比，返回相似度
#### 请求方式
- POST

#### 请求URL
    http://{host}:{port}/face/identity/compare
#### Headers
    Content-Type  application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|image_a|String|是|证件头像图片的base64编码字符串，宽:102，高:126|
|image_b|String|是|实际场景抓拍图片的base64编码字符串，宽：480，高：640|
|threshold|float|否|人脸检测的阈值，0~1，默认0.95|
|min_size|integer|否|最小的人脸尺寸，30~300，默认40|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，没检测到人脸也会返回错误，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|face_coord|数组|image_b的人脸位置坐标，（左上角x, 左上角y,右下角x, 右下角y）|
|mask|bool| true 为戴口罩，false为未戴口罩|
|similar_score|float|相似度，0到1之间，大于0.86 可认为是同一个人|

#### 请求示例
~~~http
POST /face/identity/compare   HTTP/1.1
Host: 183.252.15.157:7006
Content-Type: application/json

{
  "image_a": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe...",
  "image_b": "/9i/dnsuihfiqwhriasjfiojweifjh8y458yihe..."
  "threshold": 0.95,
  "min_size": 40
}
~~~

#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
        "face_coord": [10, 10, 110, 110],
        "simliar_score": 0.90,
        "mask": false
    }
  ]
}
~~~