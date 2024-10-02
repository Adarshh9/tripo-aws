if __name__ == '__main__':
    from main import generate_mesh

    def handler(event, context):
        print('inside handler')
        try:
            image_path = event['data']
            print(f'Image path: {image_path}')
            s3_url = generate_mesh(image_path=image_path, output_dir='tmp/output/')
            print(f'S3 URL: {s3_url}')
            return {'s3_url': s3_url}
        except Exception as e:
            print(f'Error in handler: {e}')
            return {'error': str(e)}
