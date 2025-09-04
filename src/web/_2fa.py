import pymysql
from pymysql import cursors
from web.service import address_alignment


# 函数定义

def query_parent(region_type_id, region_name,add_text=None):
    MYSQL_CONFIG = {
        'host': '127.0.0.1',
        'port': 3306,
        'user': 'root',
        'password': 'root',
        'database': 'address',
        'charset': 'utf8mb4'
    }
    with pymysql.connect(**MYSQL_CONFIG) as conn:
        with conn.cursor(cursors.DictCursor) as cursor:
            sql = (
                "select "
                "region_parent.id as parent_id, "
                "region_parent.name as parent_name, "
                "region.name as name, "
                "region.full_name "
                "from region "
                "join region as region_parent on region_parent.id=region.parent_id "
                "where region.region_type=%s and region.name like %s"
            )
            filter = (region_type_id, f"%{region_name}%")
            if add_text:
                sql += " and region_parent.name like %s"
                filter = (region_type_id, f"%{region_name}%", f"%{add_text}%")
            cursor.execute(sql, filter)
            result = cursor.fetchall()
            return result


def filter_data(result):
    result = {
        "2": result['省份'],
        "3": result['城市'],
        "4": result['区县'],
        "5": result['街道'],
        "6": result['详细地址']
    }

    if result['5']:
        query_data = query_parent(5, result['5'])
        if len(query_data) > 1 and result['4']:
            query_data = query_parent(5, result['5'][:-1], result['4'][-3:])
        if query_data and query_data[0]['name']:
            result['5'] = query_data[0]['name']
            if query_data[0]['parent_name'] != result['4']:
                result['4'] = query_data[0]['parent_name']
        else:
            result['5'] = None
    if result['4']:
        query_data = query_parent(4, result['4'])
        # if len(query_data) > 1 and result['5']:
        #     query_data = query_parent(4, result['4'], result['3'][-2:])
        if query_data and query_data[0]['name']:
            result['4'] = query_data[0]['name']
            if query_data[0]['parent_name'] != result['3']:
                result['3'] = query_data[0]['parent_name']
        else:
            result['4'] = None
    if result['3']:
        query_data = query_parent(3, result['3'])
        # if len(query_data) > 1 and result['2']:
        #     query_data = query_parent(3, result['3'], result['2'][-3:])
        if query_data and query_data[0]['name']:
            result['3'] = query_data[0]['name']
            if query_data[0]['parent_name'] != result['2']:
                result['2'] = query_data[0]['parent_name']
        else:
            result['3'] = None
    if result['2']:
        query_data = query_parent(2, result['2'])
        if len(query_data) > 1 and result['5']:
            query_data = query_parent(2, result['2'], result['5'][-3:])
        if query_data and query_data[0]['name']:
            result['2'] = query_data[0]['name']
        else:
            result['2'] = None

    data = {
        "省份": result['2'],
        "城市": result['3'],
        "区县": result['4'],
        "街道": result['5'],
        "详细地址": result['6']
    }
    return data

# # 案例2：指定区域ID=1001，进一步过滤
# parent_data = query_parent(1, "北京", 1001)
# print(parent_data)  # 仅返回 region.id=1001 且 name="北京" 的父区域数据

# if __name__ == '__main__':
#     filter_data()
